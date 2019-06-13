import os
import math
import torch
import torch.utils.data
import time
from kge.job import Job
from kge.model import KgeModel
from kge.util import KgeLoss, KgeOptimizer
import kge.job.util


class TrainingJob(Job):
    """Job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    """

    def __init__(self, config, dataset, parent_job=None):
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        self.model = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.loss = KgeLoss.create(config)
        self.batch_size = config.get("train.batch_size")
        self.device = self.config.get("job.device")
        valid_conf = config.clone()
        valid_conf.set("job.type", "eval")
        valid_conf.set("eval.data", "valid")
        valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
        self.valid_job = EvaluationJob.create(
            valid_conf, dataset, parent_job=self, model=self.model
        )
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch = self.config.get("train.trace_level") == "batch"
        self.epoch = 0
        self.valid_trace = []
        self.model.train()

        #: Hooks run after training for an epoch. Takes this job and epoch trace entry
        # as input.
        self.post_epoch_hooks = []

        #: Hooks run before outputting the trace of an batch. Takes this job and batch
        # trace entry as input and can modify the latter. Only executed when trace level
        # is batch.
        self.post_batch_update_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Takes this job and epoch
        # trace entry as input and can modify the latter.
        self.post_epoch_update_trace_hooks = []

        #: Hooks run after a validation job. Takes this job and a valid trace entry as
        # input.
        self.post_valid_hooks = []

        # let the model add some hooks, if it wants to do so
        self.model.prepare_training_job(self)

    def create(config, dataset, parent_job=None):
        """Factory method to create a training job and add necessary label_coords to
the dataset (if not present).

        """
        if config.get("train.type") == "1toN":
            return TrainingJob1toN(config, dataset, parent_job)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run_epoch(self) -> dict:
        "Runs an epoch and returns a trace entry."
        raise NotImplementedError

    def run(self):
        self.config.log("Starting training...")
        checkpoint = self.config.get("checkpoint.every")
        metric_name = self.config.get("valid.metric")
        early_stopping = self.config.get("valid.early_stopping")
        while True:
            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break
            if early_stopping > 0 and len(self.valid_trace) > early_stopping:
                stop_now = True
                last = self.valid_trace[-1][metric_name]
                for i in range(early_stopping):
                    if last > self.valid_trace[-2 - i][metric_name]:
                        stop_now = False
                        break
                if stop_now:
                    self.config.log(
                        (
                            "Stopping early ({} did not improve "
                            + "in the last {} validation runs)."
                        ).format(metric_name, early_stopping)
                    )
                    break

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            for f in self.post_epoch_hooks:
                f(self, trace_entry)
            self.config.log("Finished epoch {}.".format(self.epoch))

            # validate
            if (
                self.config.get("valid.every") > 0
                and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self, trace_entry)

            # create checkpoint and delete old one, if necessary
            self.save(self.config.checkpoint_file(self.epoch))
            if self.epoch > 1:
                if not (checkpoint > 0 and ((self.epoch - 1) % checkpoint == 0)):
                    self.config.log(
                        "Removing old checkpoint {}...".format(
                            self.config.checkpoint_file(self.epoch - 1)
                        )
                    )
                    os.remove(self.config.checkpoint_file(self.epoch - 1))

    def save(self, filename):
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()

    def resume(self):
        last_checkpoint = self.config.last_checkpoint()
        if last_checkpoint is not None:
            checkpoint_file = self.config.checkpoint_file(last_checkpoint)
            self.load(checkpoint_file)
        else:
            self.config.log("No checkpoint found, starting from scratch...")


class TrainingJob1toN(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)

        config.log("Initializing 1-to-N training job...")
        self.is_prepared = False

        # TODO currently assuming BCE loss
        self.config.check("train.loss", ["bce"])

    def _prepare(self):
        """Construct all indexes needed to run an epoch."""

        if self.is_prepared:
            return

        # create sp and po label_coords (if not done before)
        train_sp = self.dataset.index_1toN("train", "sp")
        train_po = self.dataset.index_1toN("train", "po")

        # convert indexes to pytoch tensors: a nx2 keys tensor (rows = keys),
        # an offset vector (row = starting offset in values for corresponding
        # key), a values vector (entries correspond to values of original
        # index)
        #
        # Afterwards, it holds:
        # index[keys[i]] = values[offsets[i]:offsets[i+1]]
        def prepare_index(index):
            keys = torch.tensor(list(index.keys()), dtype=torch.int)
            values = torch.cat(list(index.values()))
            offsets = torch.cumsum(
                torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
            )
            return keys, values, offsets

        self.train_sp_keys, self.train_sp_values, self.train_sp_offsets = prepare_index(
            train_sp
        )
        self.train_po_keys, self.train_po_values, self.train_po_offsets = prepare_index(
            train_po
        )

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(len(train_sp) + len(train_po)),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.num_examples = len(train_sp) + len(train_po)
        self.is_prepared = True

    def _get_collate_fun(self):
        num_sp = len(self.train_sp_keys)

        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a triple of:

            - pairs (nx2 tensor, row = sp or po indexes),
            - label coordinates (position of ones in a batch_size x num_entities tensor)
            - is_sp (vector of size n, 1 if corresponding example_index is sp, 0 if po)

            """
            # count how many labels we have
            num_ones = 0
            for example_index in batch:
                if example_index < num_sp:
                    num_ones += self.train_sp_offsets[example_index + 1]
                    num_ones -= self.train_sp_offsets[example_index]
                else:
                    example_index -= num_sp
                    num_ones += self.train_po_offsets[example_index + 1]
                    num_ones -= self.train_po_offsets[example_index]

            # now create the results
            pairs = torch.zeros([len(batch), 2], dtype=torch.long)
            is_sp = torch.zeros([len(batch)], dtype=torch.long)
            label_coords = torch.zeros([num_ones, 2], dtype=torch.long)
            current_index = 0
            for batch_index, example_index in enumerate(batch):
                is_sp[batch_index] = 1 if example_index < num_sp else 0
                if is_sp[batch_index]:
                    keys = self.train_sp_keys
                    offsets = self.train_sp_offsets
                    values = self.train_sp_values
                else:
                    example_index -= num_sp
                    keys = self.train_po_keys
                    offsets = self.train_po_offsets
                    values = self.train_po_values

                pairs[batch_index,] = keys[example_index]
                start = offsets[example_index]
                end = offsets[example_index + 1]
                size = end - start
                label_coords[current_index : (current_index + size), 0] = batch_index
                label_coords[current_index : (current_index + size), 1] = values[
                    start:end
                ]
                current_index += size

            # all done
            return pairs, label_coords, is_sp

        return collate

    def run_epoch(self) -> dict:
        self._prepare()

        # TODO refactor: much of this can go to TrainingJob
        sum_loss = 0.0
        sum_penalty = 0.0
        epoch_time = -time.time()

        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        for batch_index, batch in enumerate(self.loader):
            batch_prepare_time = -time.time()
            pairs = batch[0].to(self.device)
            batch_size = len(pairs)
            label_coords = batch[1].to(self.device)
            is_sp = batch[2]
            sp_indexes = is_sp.nonzero().to(self.device).view(-1)
            po_indexes = (is_sp == 0).nonzero().to(self.device).view(-1)
            labels = kge.job.util.coord_to_sparse_tensor(
                batch_size, self.dataset.num_entities, label_coords, self.device
            ).to_dense()
            batch_prepare_time += time.time()
            prepare_time += batch_prepare_time

            # forward pass
            batch_forward_time = -time.time()
            self.optimizer.zero_grad()
            loss_value = torch.zeros(1, device=self.device)
            penalty_value = torch.zeros(1, device=self.device)
            if len(sp_indexes) > 0:
                scores_sp = self.model.score_sp(
                    pairs[sp_indexes, 0], pairs[sp_indexes, 1]
                )
                loss_value = loss_value + self.loss(
                    scores_sp.view(-1), labels[sp_indexes,].view(-1)
                )
            if len(po_indexes) > 0:
                scores_po = self.model.score_po(
                    pairs[po_indexes, 0], pairs[po_indexes, 1]
                )
                loss_value = loss_value + self.loss(
                    scores_po.view(-1), labels[po_indexes,].view(-1)
                )
            sum_loss += loss_value.item() * batch_size
            penalty_values = self.model.penalty(
                self.epoch, batch_index, len(self.loader)
            )
            for penalty_value in penalty_values:
                penalty_value = loss_value + penalty_value
            sum_penalty += penalty_value.item()
            batch_forward_time += time.time()
            forward_time += batch_forward_time

            # backward pass
            batch_backward_time = -time.time()
            cost_value = loss_value + penalty_value
            cost_value.backward()
            batch_backward_time += time.time()
            backward_time += batch_backward_time

            # upgrades
            batch_optimizer_time = -time.time()
            self.optimizer.step()
            batch_optimizer_time += time.time()
            optimizer_time += batch_optimizer_time

            if self.trace_batch:
                batch_trace = {
                    "type": "1toN",
                    "scope": "batch",
                    "epoch": self.epoch,
                    "batch": batch_index,
                    "size": batch_size,
                    "batches": len(self.loader),
                    "avg_loss": loss_value.item(),
                    "penalties": [p.item() for p in penalty_values],
                    "penalty": penalty_value.item(),
                    "cost": cost_value.item(),
                    "prepare_time": batch_prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                }
                for f in self.post_batch_update_trace_hooks:
                    f(self, batch_trace)
                self.trace(**batch_trace)
            print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    loss_value.item(),
                    penalty_value.item(),
                    cost_value.item(),
                    batch_prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

        epoch_time += time.time()
        print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )
        trace_entry = {
            "echo": True,
            "echo_prefix": "  ",
            "log": True,
            "type": "1toN",
            "scope": "epoch",
            "epoch": self.epoch,
            "batches": len(self.loader),
            "size": self.num_examples,
            "avg_loss": sum_loss / self.num_examples,
            "avg_penalty": sum_penalty / len(self.loader),
            "avg_cost": sum_loss / self.num_examples + sum_penalty / len(self.loader),
            "epoch_time": epoch_time,
            "prepare_time": prepare_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "other_time": other_time,
        }
        for f in self.post_epoch_update_trace_hooks:
            f(self, trace_entry)
        trace_entry = self.trace(**trace_entry)
        return trace_entry
