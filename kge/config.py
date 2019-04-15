import collections
import copy
import datetime
import os
import time
import yaml
import uuid


from kge.util.misc import is_number


class Config:
    """Configuration options.

    All available options, their types, and their descriptions are defined in
    :file:`config_default.yaml`.
    """

    def __init__(self):
        """Initialize with the default configuration"""
        with open('kge/config-default.yaml', 'r') as file:
            self.options = yaml.load(file, Loader=yaml.SafeLoader)

    # -- ACCESS METHODS -------------------------------------------------------

    def get(self, key, remove_plusplusplus=True):
        """Obtain value of specified key.

        Nested dictionary values can be accessed via "." (e.g.,
        "output.folder"). Strips all '+++' keys unless `remove_plusplusplus` is
        set to `False`.

        """
        result = self.options
        for name in key.split('.'):
            result = result[name]

        if remove_plusplusplus and isinstance(result, collections.Mapping):
            def do_remove_plusplusplus(option):
                if isinstance(option, collections.Mapping):
                    option.pop('+++', None)
                    for values in option.values():
                        do_remove_plusplusplus(values)

            result = copy.deepcopy(result)
            do_remove_plusplusplus(result)

        return result

    def set(self, key, value, create=False):
        """Set value of specified key.

        Nested dictionary values can be accessed via "." (e.g.,
        "output.folder").

        If ``create`` is ``False``, raises :class:`ValueError` when the key
        does not exist already; otherwise, the new key-value pair is inserted
        into the configuration.

        """
        splits = key.split('.')
        data = self.options

        path = []
        for i in range(len(splits) - 1):
            create = create or '+++' in data[splits[i]]
            if create and splits[i] not in data:
                data[splits[i]] = dict()
            path.append(splits[i])
            data = data[splits[i]]

        current_value = data.get(splits[-1])
        if current_value is None:
            if not create:
                raise ValueError("key {} not present".format(key))
        elif type(value) != type(current_value):
            raise ValueError("key {} has incorrect type".format(key))

        if isinstance(value, str) and is_number(value, float):
            value = float(value)
        elif isinstance(value, str) and is_number(value, int):
            value = int(value)

        data[splits[-1]] = value
        return value

    def set_all(self, new_options, create=False, detect_model_config=False):
        found_model_config = False
        for key, value in Config.flatten(new_options).items():
            if detect_model_config and key == 'model.type':
                self.load('kge/model/{}.yaml'.format(value), create=True,
                          detect_model_config=False)
                found_model_config = True
            self.set(key, value, create)
        if detect_model_config and not found_model_config:
            self.load('kge/model/{}.yaml'.format(self.get('model.type')),
                      create=True, detect_model_config=False)

    def load(self, filename, create=False, detect_model_config=True):
        """Update configuration options from the specified YAML file.

        All options that do not occur in the specified file are retained.

        If ``create`` is ``False``, raises :class:`ValueError` when the file
        contains a non-existing options. When ``create`` is ``False``, allows
        to add options that are not present in this configuration.

        """
        with open(filename, 'r') as file:
            new_options = yaml.load(file, Loader=yaml.SafeLoader)
        self.set_all(new_options, create, detect_model_config)

    def save(self, filename):
        """Save this configuration to the given file"""
        with open(filename, "w+") as file:
            file.write(yaml.dump(self.options))

    @staticmethod
    def flatten(options):
        """Returns a dictionary of flattened configuration options."""
        result = {}
        Config.__flatten(options, result)
        return result

    @staticmethod
    def __flatten(options, result, prefix=''):
        for key, value in options.items():
            fullkey = key if prefix == '' else prefix + '.' + key
            if type(value) is dict:
                Config.__flatten(value, result, prefix=fullkey)
            else:
                result[fullkey] = value

    def clone(self, subfolder=None):
        """Return a deep copy"""
        new_config = copy.deepcopy(self)
        if subfolder is not None:
            new_config.set("output.folder", self.folder() + subfolder + '/')
        return new_config

    # -- LOGGING AND TRACING --------------------------------------------------

    def log(self, msg, echo=True, prefix=''):
        """Add a message to the default log file.

        Optionally also print on console. ``prefix`` is used to indent each
        output line.

        """
        with open(self.logfile(), 'a') as file:
            for line in msg.splitlines():
                if prefix:
                    line = prefix + line
                if echo:
                    print(line)
                file.write(str(datetime.datetime.now()))
                file.write(" ")
                file.write(line)
                file.write("\n")

    def trace(self, echo=False, echo_prefix='', echo_flow=False,
              log=False, **kwargs):
        """Write a set of key-value pairs to the trace file.

        The pairs are written as a single-line YAML record. Optionally, also
        echo to console and/or write to log file.

        And id and the current time is automatically added using key ``timestamp``.
        """
        with open(self.tracefile(), 'a') as file:
            kwargs['timestamp'] = time.time()
            kwargs['entry_id'] = str(uuid.uuid4())
            line = yaml.dump(
                kwargs, width=float('inf'), default_flow_style=True).strip()
            if echo or log:
                msg = yaml.dump(kwargs, default_flow_style=echo_flow)
                if log:
                    self.log(msg, echo, echo_prefix)
                else:
                    for line in msg.splitlines():
                        if echo_prefix:
                            line = echo_prefix + line
                            print(line)
            file.write(line)
            file.write("\n")
            return line

    # -- FOLDERS AND CHECKPOINTS ----------------------------------------------

    def init_folder(self):
        """Initialize the output folder.

        If the folder does not exists, create it, dump the configuration
        there and return ``True``. Else do nothing and return ``False``.

        """
        if not os.path.exists(self.folder()):
            os.makedirs(self.folder())
            self.save(self.folder() + "/config.yaml")
            return True
        return False

    def checkpointfile(self, epoch):
        "Return path of checkpoint file for given epoch"
        return "{}/{}_{:05d}.pt".format(
            self.folder(), self.get('checkpoint.basefile'), epoch)

    def last_checkpointfile(self):
        "Return name of latest checkpoint file"
        # stupid implementation, but works
        tried_epoch = 0
        found_epoch = 0
        while tried_epoch < found_epoch + 100:
            tried_epoch += 1
            if os.path.exists(self.checkpointfile(tried_epoch)):
                found_epoch = tried_epoch
        if found_epoch > 0:
            return self.checkpointfile(found_epoch)
        else:
            return None

    # -- CONVENIENCE METHODS --------------------------------------------------

    def check(self, key, allowed_values):
        """Raise an error if value of key is not in allowed"""
        if not self.get(key) in allowed_values:
            raise ValueError(
                "Illegal value {} for key {}; allowed values are {}"
                .format(self.get(key), key, allowed_values))

    def folder(self):
        """Return output folder"""
        folder = self.get('output.folder')
        if len(folder) > 0 and not folder.endswith('/'):
            folder += '/'
        return folder

    def logfile(self):
        return self.folder() + self.get('output.logfile')

    def tracefile(self):
        return self.folder() + self.get('output.tracefile')
