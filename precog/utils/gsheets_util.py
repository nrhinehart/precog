from abc import ABCMeta, abstractmethod
import inspect
from functools import wraps
import logging
import os
import pdb
import six

import precog.utils.class_util as classu

log = logging.getLogger(__file__)

try:
    import pygsheets
    # You'll need your own one of these!
    secret_filename = os.path.dirname(__file__) + '/client_secret.json'
    if not os.path.isfile(secret_filename):
        log.warning("You haven't set up your google-sheets login! See the `pygsheets` repo's README")
    client_singleton = pygsheets.authorize(client_secret=secret_filename)
    try:
        @six.add_metaclass(ABCMeta)
        class GSheetsResults:
            # This decorator automatically sets member attributes of the __init__ args. 
            @classu.member_initialize
            def __init__(self, sheet_name, worksheet_name='Sheet1', client=client_singleton):
                try:
                    self.sheet = self.client.open(self.sheet_name)
                except pygsheets.exceptions.SpreadsheetNotFound:
                    self.sheet = self.client.create(self.sheet_name)
                try:
                    self.wks = self.sheet.worksheet_by_title(self.worksheet_name)
                except pygsheets.exceptions.WorksheetNotFound:
                    self.wks = self.sheet.add_worksheet(self.worksheet_name, index=1)
                self.allocated_row = False
                self.row_claim_tag = None

            def __len__(self):
                count = 0
                for row in self.wks:
                    count += 1
                return count

            def claim_row(self, tag):
                """Claim a row with a unique tag

                :param tag: str unique tag
                :param row_index: optional index 
                :returns: 
                :rtype: 

                """
                assert(self.row_claim_tag is None)
                if len(self.wks.find(tag)) > 0:
                    raise ValueError("Cannot claim row with existing tag: '{}'".format(tag))
                self.row_claim_tag = tag
                self.wks.insert_rows(row=len(self), number=1, values=[[tag]])

            def update_claimed_row(self, row_data):
                """Update the claimed row with data.

                :param row_data: list of data to put in the row.
                :returns: 
                :rtype: 

                """
                assert(isinstance(row_data, list))
                assert(self.row_claim_tag is not None)
                matches = self.wks.find(self.row_claim_tag)
                if len(matches) == 0:
                    raise ValueError("Can't find tag!")
                elif len(matches) == 0:
                    pass
                else:
                    log.warning("Tag matched multiple cells! Filtering to column-1 cells.")
                    col1_matches = [_ for _ in matches if _.col == 1]
                    # If there are multiple column-1 matches, we shouldn't write anything anywhere.
                    assert(len(col1_matches) == 1)
                    row = col1_matches[0].row
                    # Ensure all of the duplicates occur in the same row.
                    assert(all([row == _.row for _ in matches]))
                    # Set the matches to just be the length-1 list of first column match cell.
                    matches = col1_matches

                assert(len(matches) == 1)
                # The updated values always include the tag.
                self.wks.update_values(crange='A{}'.format(matches[0].row), values=[[self.row_claim_tag] + row_data], extend=True)
        have_pygsheets = True
    except ImportError:
        have_pygsheets = False
except FileNotFoundError:
    have_pygsheets = False
