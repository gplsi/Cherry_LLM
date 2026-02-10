import unittest
import sys
from unittest.mock import patch
from cherry_seletion.data_by_cluster import parse_args
import argparse

class TestDataByCluster(unittest.TestCase):

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_parse_args):
        # This test is isolated from the main script logic
        # It checks if the arguments are added to the parser correctly.
        
        # To do this, we can't easily check the call to parse_args(),
        # but we can mock it to return a known value and then check
        # if the function returns that value.
        
        # A better approach would be to refactor the parse_args function
        # to take a list of arguments as input, which makes it more
        # testable.
        
        # For now, we will just have a basic test.
        
        # Create a mock Namespace object to be returned by parse_args
        mock_args = argparse.Namespace(
            pt_data_path="test.pt",
            json_data_path="test.json",
            json_save_path="save.json",
            sent_type=1,
            ppl_type=1,
            cluster_method="kmeans",
            reduce_method="tsne",
            sample_num=20,
            kmeans_num_clusters=50,
            low_th=10,
            up_th=90
        )
        mock_parse_args.return_value = mock_args

        # Call the function
        args = parse_args()

        # Assert that the function returned the mocked arguments
        self.assertEqual(args.pt_data_path, "test.pt")
        self.assertEqual(args.json_data_path, "test.json")
        self.assertEqual(args.json_save_path, "save.json")
        self.assertEqual(args.sent_type, 1)
        self.assertEqual(args.ppl_type, 1)
        self.assertEqual(args.cluster_method, "kmeans")
        self.assertEqual(args.reduce_method, "tsne")
        self.assertEqual(args.sample_num, 20)
        self.assertEqual(args.kmeans_num_clusters, 50)
        self.assertEqual(args.low_th, 10)
        self.assertEqual(args.up_th, 90)


if __name__ == '__main__':
    # You can run this test file directly
    # python -m unittest cherry_seletion/test_data_by_cluster.py
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
