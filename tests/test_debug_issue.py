#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Demonstrate pathological debug issue.
"""
import unittest

import parlai.utils.logging as logging
import parlai.utils.testing as testing_utils


class Test1(unittest.TestCase):
    def test1(self):
        logging.set_log_level('DEBUG')
        with self.assertLogs(logger=logging.logger, level='DEBUG') as cm:
            logging.debug('foo')
            assert any('foo' in l for l in cm.output)


class Test2(unittest.TestCase):
    def test2(self):
        logging.set_log_level('INFO')
        logging.debug('foo')


    def test_no_greedy_largebeam(self):
        """
        Ensures that --beam-size > 1 and --inference greedy causes a failure.
        """
        # # we should have an exception if we mix beam size > 1 with inference greedy
        # with self.assertRaises(ValueError):
        #     testing_utils.display_model(
        #         dict(
        #             task='integration_tests:multiturn_nocandidate',
        #             model_file='zoo:unittest/transformer_generator2/model',
        #             beam_size=5,
        #             inference='greedy',
        #         ), skip_capture=True
        #     )

        # and we shouldn't if we have inference beam
        with self.assertLogs(logger=logging.logger, level='INFO') as cm:
            testing_utils.display_model(
                dict(
                    task='integration_tests:multiturn_nocandidate',
                    model_file='zoo:unittest/transformer_generator2/model',
                    beam_size=5,
                    inference='beam'
                ), skip_capture=True
            )


class Test3(unittest.TestCase):
    def test3(self):
        logging.set_log_level('DEBUG')
        with self.assertLogs(logger=logging.logger, level='DEBUG') as cm:
            logging.debug('foo')
            assert any('foo' in l for l in cm.output)


if __name__ == '__main__':
    unittest.main()
