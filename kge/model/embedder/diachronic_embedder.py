from torch import Tensor
from torch._C import dtype
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


class DiachronicEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.se_prop = self.get_option("se_prop")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size
        self.time_size = self.get_option("time_size")

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        self.t_emb_dim = self.dim - int(self.se_prop * self.dim)
        self.s_emb_dim = int(self.se_prop * self.dim)

        self._ent_emb = torch.nn.Embedding(
            self.vocab_size, self.s_emb_dim, sparse=self.sparse
        )
        self._freq_emb = torch.nn.Embedding(
            self.vocab_size, self.t_emb_dim, sparse=self.sparse
        )
        self._phi_emb = torch.nn.Embedding(
            self.vocab_size, self.t_emb_dim, sparse=self.sparse
        )
        self._amp_emb = torch.nn.Embedding(
            self.vocab_size, self.t_emb_dim, sparse=self.sparse
        )

        if not init_for_load_only:
            # initialize weights
            self.initialize(self._ent_emb.weight.data)
            self.initialize(self._freq_emb.weight.data)
            self.initialize(self._phi_emb.weight.data)
            self.initialize(self._amp_emb.weight.data)
            self._normalize_embeddings()

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0 and config.get("train.auto_correct"):
            config.log(
                "Setting {}.dropout to 0., "
                "was set to {}.".format(configuration_key, dropout)
            )
            dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def _normalize_embeddings(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                self._ent_emb.weight.data = torch.nn.functional.normalize(
                    self._freq_emb.weight.data, p=self.normalize_p, dim=-1
                )
                self._freq_emb.weight.data = torch.nn.functional.normalize(
                    self._freq_emb.weight.data, p=self.normalize_p, dim=-1
                )
                self._phi_emb.weight.data = torch.nn.functional.normalize(
                    self._freq_emb.weight.data, p=self.normalize_p, dim=-1
                )
                self._amp_emb.weight.data = torch.nn.functional.normalize(
                    self._freq_emb.weight.data, p=self.normalize_p, dim=-1
                )

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0 and isinstance(job, TrainingJob):
            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self._normalize_embeddings())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self._normalize_embeddings())

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: "DiachronicEmbedder") -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        self._ent_emb.weight[
            torch.from_numpy(self_intersect_ind).to(self._ent_emb.weight.device).long()
        ] = pretrained_embedder._postprocess(
            pretrained_embedder._ent_emb(
                torch.from_numpy(pretrained_intersect_ind).to(
                    self._ent_emb.weight.device
                )
            )
        )
        self._freq_emb.weight[
            torch.from_numpy(self_intersect_ind).to(self._freq_emb.weight.device).long()
        ] = pretrained_embedder._postprocess(
            pretrained_embedder._freq_emb(
                torch.from_numpy(pretrained_intersect_ind).to(
                    self._freq_emb.weight.device
                )
            )
        )
        self._phi_emb.weight[
            torch.from_numpy(self_intersect_ind).to(self._phi_emb.weight.device).long()
        ] = pretrained_embedder._postprocess(
            pretrained_embedder._phi_emb(
                torch.from_numpy(pretrained_intersect_ind).to(
                    self._phi_emb.weight.device
                )
            )
        )
        self._amp_emb.weight[
            torch.from_numpy(self_intersect_ind).to(self._amp_emb.weight.device).long()
        ] = pretrained_embedder._postprocess(
            pretrained_embedder._amp_emb(
                torch.from_numpy(pretrained_intersect_ind).to(
                    self._amp_emb.weight.device
                )
            )
        )

    def _embed(self, indexes: Tensor, time: Tensor) -> Tensor:
        # TODO: add other activations than sin
        return torch.cat(
            (
                self._ent_emb(indexes),
                (
                    self._amp_emb(indexes)
                    * torch.sin(
                        self._freq_emb(indexes) * time.view(-1, 1)
                        + self._phi_emb(indexes)
                    )
                ),
            ),
            1,
        )

    def embed(self, indexes: Tensor, time: Tensor) -> Tensor:
        return self._postprocess(self._embed(indexes, time))

    def embed_all(self) -> Tensor:
        return self._postprocess(self._embeddings_all())

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        return self._embed(
            torch.arange(
                self.vocab_size, dtype=torch.long, device=self._freq_emb.weight.device
            ),
            torch.arange(
                self.time_size, dtype=torch.long, device=self._freq_emb.weight.device
            ),
        )

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embed(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
