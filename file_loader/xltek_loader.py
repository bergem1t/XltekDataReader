import os
import numpy as np
from collections import OrderedDict
from .utils.misc import frame_filetime

from .file_types.eeg_file import EEGLoader
from .file_types.erd_file import ERDLoader
from .file_types.ent_file import ENTLoader
from .file_types.etc_file import ETCLoader
from .file_types.snc_file import SNCLoader
from .file_types.stc_file import STCLoader
from .file_types.vtc_file import VTCLoader

loaders = OrderedDict(
    eeg=[EEGLoader, "eeg_loader"],
    erd=[ERDLoader, "erd_loader_{}"],
    ent=[ENTLoader, "ent_loader"],
    etc=[ETCLoader, "etc_loader_{}"],
    snc=[SNCLoader, "snc_loader"],
    stc=[STCLoader, "stc_loader"],
    vtc=[VTCLoader, "vtc_loader"]
)

class XltekLoader:
    def __init__(self, load_dir):
        self.load_dir = load_dir
        self.files_dict = dict()
        self.loaders_dict = OrderedDict()

        # First scan for files of interest
        for f in os.listdir(load_dir):
            f_full = os.path.join(load_dir, f)
            if os.path.isfile(f_full):
                file_split_lst = f.split('.')
                if file_split_lst[-2].endswith('new'): continue
                file_type = file_split_lst[-1]
                if file_type in loaders.keys():
                    if file_type not in self.files_dict.keys():
                        self.files_dict[file_type] = []
                    self.files_dict[file_type].append(f_full)
        
        # sort it
        for k,v in self.files_dict.items():
          self.files_dict[k] = list(np.sort(v))

        # Assign file loaders for each of the files
        for file_type in loaders.keys():
            for f_id, f in enumerate(self.files_dict[file_type]):
                loader_type, loader_name = loaders[file_type]
                self.loaders_dict[loader_name.format(f_id)] = loader_type(f)

    def combine_files(self):
        total_samples = -1
        lst_data = []

        # Get sequences and concat together
        for loader_id in range(len(self.files_dict['etc'])):

            erd_loader = self.loaders_dict['erd_loader_{}'.format(loader_id)]
            etc_loader = self.loaders_dict['etc_loader_{}'.format(loader_id)]
            packets = erd_loader.data['data_packets']

            # TODO: Find the length of this array and preallocate
            data_file = []

            # Assign sample stamp for each packet
            for t, val in enumerate(packets.values_list):
                slice_data = np.zeros((erd_loader.num_channels+1,1))

                for channel_id in range(erd_loader.num_channels):
                    # If the channel is not included in some packet,
                    # mark as nan. Interpolate them later.
                    try:
                        channel_val_idx = packets.channels_list[t].index(channel_id)
                        slice_data[channel_id] = val[channel_val_idx]
                    except ValueError as e:
                        if "is not in list" in str(e):
                            slice_data[channel_id] = np.NAN

                # If offset is not in their respective toc file,
                # assume it's just last sample stamp + 1
                try:
                    offset = packets.values_list[t]
                    offset_idx = etc_loader.data['table_of_content']['offset'].index(offset)
                    sample_stamp = etc_loader.data['samplestamp'].index(offset_idx)
                except ValueError as e:
                    if "is not in list" in str(e):
                        sample_stamp = total_samples + 1
                total_samples  = sample_stamp

                slice_data[-1, 0] = sample_stamp
                data_file.append(slice_data)
            
            lst_data.append(
                np.concatenate(data_file, axis=1)
            )

        # TODO: Make sure the names are sorted?

        # Concat them all in time
        data_array = np.concatenate(lst_data, axis=1)
        # TODO: Is this always possible?

        # Use snc to create FILETIME for each packet
        samplestamp = self.loaders_dict['snc_loader'].data['time_mappings']['samplestamp']
        filetime = self.loaders_dict['snc_loader'].data['time_mappings']['sample_time']

        # Use start & end filetime of video to interpolate each frame's filetime
        # frame_filetime_lst = frame_filetime(
        #     self.loaders_dict['vtc_loader'],
        #     self.load_dir  
        # )

        # Attach note to their sample stamp
        note_dict = dict()
        note_packets = self.loaders_dict['ent_loader'].data['note_packets']
        for tree in note_packets['note_key_tree']:
            if isinstance(tree, dict):
                note_dict[int(tree['Stamp'])] = tree
                # TODO: Handle failures in casting?

        # Return channel names for the array (plus sample stamp). 
        # TODO: Check that the channel orders/presence the same for different files
        c_names = self.loaders_dict['erd_loader_0'].channel_names

        # Return list
        ret_val = {
            'StudyInfo': self.loaders_dict['eeg_loader'].data['study_info'],
            'EEGData': data_array,
            'ChannelNames': c_names+['SampleStamp'],
            'Notes': note_dict,
            'FrameAndFiletime': [],#frame_filetime_lst,
            'FiletimeStampConversion': (
                filetime,
                samplestamp
            )
        }

        return ret_val

    def load(self):
        self.read()
        self.validate()
        res = self.combine_files()
        return res

    def read(self):
        # takes a minute or so to even load a single erd file
        # probably because of the slow VPN network
        for loader_id in range(len(self.files_dict['erd'])): # the actual data
            self.loaders_dict['erd_loader_{}'.format(loader_id)].load()
        # etc loads pretty quickly
        for loader_id in range(len(self.files_dict['etc'])): # datachunk info from erd?
            self.loaders_dict['etc_loader_{}'.format(loader_id)].load()
        self.loaders_dict['eeg_loader'].load()
        self.loaders_dict['ent_loader'].load()
        self.loaders_dict['stc_loader'].load()
        #self.loaders_dict['vtc_loader'].load() # see below
        self.loaders_dict['snc_loader'].load()

    def validate(self):
      # strainge implementation: I'm not sure what the true or false result
      # from the validate function means since even a false validation seems
      # to produce results. Howver, it produces an error if the file is not loaded
      for loader_id in range(len(self.files_dict['erd'])):
          self.loaders_dict['erd_loader_{}'.format(loader_id)].validate()
      for loader_id in range(len(self.files_dict['etc'])):
          self.loaders_dict['etc_loader_{}'.format(loader_id)].validate(
              self.loaders_dict['erd_loader_{}'.format(loader_id)]
          )  # Use toc file to validate erd
      self.loaders_dict['eeg_loader'].validate()
      self.loaders_dict['ent_loader'].validate()
      self.loaders_dict['stc_loader'].validate()
      self.loaders_dict['vtc_loader'].validate()
      self.loaders_dict['snc_loader'].validate()
        
    # new function to just read the header information without the data
    def load_header(self):
        self.loaders_dict['eeg_loader'].load() # study info - lots of it
        self.loaders_dict['ent_loader'].load() # notes
        self.loaders_dict['stc_loader'].load() # ??
        #self.loaders_dict['vtc_loader'].load() 
        #   File "c:\users\michael\gitrepos\xltekdatareader\file_loader\utils\byte_buffer.py", line 71, in read
        #     val = struct.unpack(
        
        # error: unpack requires a buffer of 16 bytes
        #
        # VTC loader should probably not be necessary since it is the video
        # we already have this information in the dayhandle
        self.loaders_dict['snc_loader'].load() # ?? timing info
        
    def validate_header(self):
      valid = True
      valid &= self.loaders_dict['eeg_loader'].validate() 
      valid &= self.loaders_dict['ent_loader'].validate() # returns false - seems to work nonetheless
      valid &= self.loaders_dict['stc_loader'].validate()
      valid &= self.loaders_dict['vtc_loader'].validate()
      valid &= self.loaders_dict['snc_loader'].validate() # returns false - seems to work nonetheless
      return valid
        
    def get_header(self):
      # Use snc to create FILETIME for each packet
      samplestamp = self.loaders_dict['snc_loader'].data['time_mappings']['samplestamp']
      filetime = self.loaders_dict['snc_loader'].data['time_mappings']['sample_time']

      # Use start & end filetime of video to interpolate each frame's filetime
      # frame_filetime_lst = frame_filetime(
      #     self.loaders_dict['vtc_loader'],
      #     self.load_dir  
      # )

      # Attach note to their sample stamp
      note_dict = dict()
      note_packets = self.loaders_dict['ent_loader'].data['note_packets']
      for tree in note_packets['note_key_tree']:
          if isinstance(tree, dict):
              note_dict[int(tree['Stamp'])] = tree
              # TODO: Handle failures in casting?

      # Return channel names for the array (plus sample stamp). 
      # TODO: Check that the channel orders/presence the same for different files
      #c_names = self.loaders_dict['erd_loader_0'].channel_names

      # Return list
      ret_val = {
          'StudyInfo': self.loaders_dict['eeg_loader'].data['study_info'],
          'EEGData': [],
          'ChannelNames': [],#c_names+['SampleStamp'],
          'Notes': note_dict,
          'FrameAndFiletime': [],#frame_filetime_lst, #-- should get this from the dayhandle
          'FiletimeStampConversion': (
              filetime,
              samplestamp
          )
      }

      return ret_val
