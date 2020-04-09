
import socket

import precog.utils.gsheets_util as gsheetsu
import precog.utils.class_util as classu

if gsheetsu.have_pygsheets:
    class ESPResults(gsheetsu.GSheetsResults):
        @classu.member_initialize
        def __init__(self, tag, dataset, output_directory):
            super().__init__(sheet_name="esp_results", worksheet_name="{}_dataset".format(dataset))
            self.claim_row(tag)
            self.metadata = [output_directory, 'host-{}'.format(socket.gethostname())]
            self.split_cache = {'train': [], 'val': [], 'test': []}

        def _fmt(self, split, args):
            return ['{} {}={:.4g}'.format(split, k, v) for k, v in args]

        def update(self, split, global_step, args):
            """Call this at a new best.

            :param split: 
            :param result_dict: 
            :returns: 
            :rtype: 

            """

            joint = ['step={}'.format(global_step)]
            self.split_cache[split] = self._fmt(split, args)
            for split in ('test', 'train', 'val'):
                joint.extend(self.split_cache[split])
            joint.extend(self.metadata)
            self.update_claimed_row(joint)
