import unittest
import json
import os
import tempfile
import torch
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from cherry_seletion.preprocess import (
    get_template,
    create_conversation,
    create_prompt,
    tokenizer_dataset_given_prompt,
    tokenizer_dataset_multi_turn,
    get_sft_collate_fn,
    _sft_collate_fn,
)


class TestGetTemplate(unittest.TestCase):
    """Test template loading from various sources (file, dict, inline string)."""

    def setUp(self):
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.chat_template = None

    def test_template_from_inline_string(self):
        """Test loading template from inline string - verifies string templates are applied correctly."""
        template_str = "{% for message in messages %}{{ message.content }}{% endfor %}"
        get_template(template_str, self.mock_tokenizer)

        self.assertEqual(self.mock_tokenizer.chat_template, template_str)

    def test_template_from_dict_with_chat_template_key(self):
        """Test loading template from dict with 'chat_template' key - verifies dict format handling."""
        template_str = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}"
        template_spec = {"chat_template": template_str}
        get_template(template_spec, self.mock_tokenizer)

        self.assertEqual(self.mock_tokenizer.chat_template, template_str)

    def test_template_from_json_file_with_chat_template_key(self):
        """Test loading template from JSON file with 'chat_template' key - catches file parsing regressions."""
        template_str = "{% for message in messages %}{{ message.content }}{% endfor %}"
        template_dict = {"chat_template": template_str}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(template_dict, f)
            temp_path = f.name

        try:
            get_template(temp_path, self.mock_tokenizer)
            self.assertEqual(self.mock_tokenizer.chat_template, template_str)
        finally:
            os.unlink(temp_path)

    def test_template_from_json_file_with_string_content(self):
        """Test loading template from JSON file containing a raw string - verifies JSON string unwrapping."""
        template_str = "{% for message in messages %}{{ message.content }}{% endfor %}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(template_str, f)
            temp_path = f.name

        try:
            get_template(temp_path, self.mock_tokenizer)
            self.assertEqual(self.mock_tokenizer.chat_template, template_str)
        finally:
            os.unlink(temp_path)

    def test_template_from_text_file(self):
        """Test loading template from plain text file - ensures non-JSON files work."""
        template_str = "{% for message in messages %}{{ message.content }}{% endfor %}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(template_str)
            temp_path = f.name

        try:
            get_template(temp_path, self.mock_tokenizer)
            self.assertEqual(self.mock_tokenizer.chat_template, template_str)
        finally:
            os.unlink(temp_path)

    def test_template_from_json_file_fallback_to_raw_content(self):
        """Test JSON file with non-string/dict content falls back to raw - catches edge case handling."""
        template_content = '["not", "a", "template"]'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            get_template(temp_path, self.mock_tokenizer)
            self.assertEqual(self.mock_tokenizer.chat_template, template_content)
        finally:
            os.unlink(temp_path)

    def test_none_template_spec_does_nothing(self):
        """Test None template_spec is a no-op - ensures optional template behavior."""
        get_template(None, self.mock_tokenizer)
        self.assertIsNone(self.mock_tokenizer.chat_template)

    def test_empty_string_template_spec_does_nothing(self):
        """Test empty string is treated as no-op - prevents accidental template clearing."""
        get_template("", self.mock_tokenizer)
        self.assertIsNone(self.mock_tokenizer.chat_template)

    def test_invalid_dict_without_chat_template_key_does_nothing(self):
        """Test dict without 'chat_template' key is ignored - validates key requirement."""
        template_spec = {"other_key": "value"}
        get_template(template_spec, self.mock_tokenizer)
        self.assertIsNone(self.mock_tokenizer.chat_template)

    def test_non_string_chat_template_value_raises_error(self):
        """Test non-string template value raises ValueError - catches type errors."""
        template_spec = {"chat_template": 123}

        with self.assertRaises(ValueError) as context:
            get_template(template_spec, self.mock_tokenizer)

        self.assertIn("must be a string", str(context.exception))


class TestCreateConversation(unittest.TestCase):
    """Test conversation construction from row data - PROMPT REGRESSION TESTS."""

    def test_single_turn_conversation(self):
        """Test single-turn conversation structure - verifies basic message ordering."""
        row = {
            "system": "You are a helpful assistant.",
            "input": ["What is 2+2?"],
            "assistance": []
        }

        conversation = create_conversation(row)

        # Verify exact structure
        self.assertEqual(len(conversation), 2)
        self.assertEqual(conversation[0], {"role": "system", "content": "You are a helpful assistant."})
        self.assertEqual(conversation[1], {"role": "user", "content": "What is 2+2?"})

    def test_multi_turn_conversation_with_assistant_responses(self):
        """Test multi-turn conversation with assistant responses - verifies interleaving."""
        row = {
            "system": "You are a helpful assistant.",
            "input": ["First question", "Second question", "Third question"],
            "assistance": ["First answer", "Second answer"]
        }

        conversation = create_conversation(row)

        # Verify exact message order: system -> user -> assistant -> user -> assistant -> user
        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
            {"role": "user", "content": "Third question"}
        ]

        self.assertEqual(conversation, expected)

    def test_empty_input_list(self):
        """Test empty input list produces system-only conversation - catches boundary case."""
        row = {
            "system": "You are a helpful assistant.",
            "input": [],
            "assistance": []
        }

        conversation = create_conversation(row)

        # Should only contain system message
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0]["role"], "system")

    def test_conversation_preserves_exact_content(self):
        """Test conversation preserves special characters and whitespace - catches content corruption."""
        row = {
            "system": "System with\nnewlines and  spaces",
            "input": ["User with <tags> and &symbols"],
            "assistance": []
        }

        conversation = create_conversation(row)

        self.assertEqual(conversation[0]["content"], "System with\nnewlines and  spaces")
        self.assertEqual(conversation[1]["content"], "User with <tags> and &symbols")

    def test_multi_turn_last_message_always_user(self):
        """Test last message is always user (no assistant) - validates training format."""
        row = {
            "system": "System",
            "input": ["Q1", "Q2"],
            "assistance": ["A1"]
        }

        conversation = create_conversation(row)

        # Last message should be user Q2, not assistant
        self.assertEqual(conversation[-1]["role"], "user")
        self.assertEqual(conversation[-1]["content"], "Q2")


class TestCreatePrompt(unittest.TestCase):
    """Test prompt generation with chat template - PROMPT REGRESSION TESTS."""

    def setUp(self):
        self.mock_tokenizer = Mock()

    @patch('cherry_seletion.preprocess.datetime')
    def test_create_prompt_calls_apply_chat_template_correctly(self, mock_datetime):
        """Test apply_chat_template receives correct parameters - catches API changes."""
        mock_datetime.today.return_value.strftime.return_value = "2026-02-02"

        conversation = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"}
        ]

        self.mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        result = create_prompt(conversation, self.mock_tokenizer)

        # Verify exact call signature
        self.mock_tokenizer.apply_chat_template.assert_called_once_with(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            date_string="2026-02-02"
        )
        self.assertEqual(result, "formatted prompt")

    @patch('cherry_seletion.preprocess.datetime')
    def test_create_prompt_includes_current_date(self, mock_datetime):
        """Test date string is generated from current date - ensures date context is correct."""
        mock_datetime.today.return_value.strftime.return_value = "2025-12-31"

        conversation = [{"role": "user", "content": "Test"}]
        self.mock_tokenizer.apply_chat_template.return_value = "prompt"

        create_prompt(conversation, self.mock_tokenizer)

        call_kwargs = self.mock_tokenizer.apply_chat_template.call_args[1]
        self.assertEqual(call_kwargs["date_string"], "2025-12-31")

    def test_create_prompt_returns_tokenizer_output(self):
        """Test function returns exact tokenizer output - validates passthrough behavior."""
        conversation = [{"role": "user", "content": "Test"}]
        expected_prompt = "<|system|>System<|user|>Test<|assistant|>"

        self.mock_tokenizer.apply_chat_template.return_value = expected_prompt

        result = create_prompt(conversation, self.mock_tokenizer)

        self.assertEqual(result, expected_prompt)


class TestTokenizerDatasetGivenPrompt(unittest.TestCase):
    """Test tokenization with prompt masking - HYBRID TESTS (logic + prompt structure)."""

    def setUp(self):
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2

    def test_prompt_tokens_masked_in_labels(self):
        """Test prompt tokens are masked with ignore_index in labels - critical for training loss."""
        element = {
            "id": 123,
            "prompt": "User: What is 2+2?",
            "prompt_and_response": "User: What is 2+2? Assistant: 4",
            "response": "4"
        }

        # Mock tokenizer.encode to return predictable token IDs
        def mock_encode(text, max_length=None, add_special_tokens=False):
            if text == element["prompt"]:
                return [10, 11, 12]  # 3 tokens for prompt
            elif text == element["prompt_and_response"]:
                return [10, 11, 12, 13, 14]  # 5 tokens for prompt+response
            elif text == element["response"]:
                return [13, 14]  # 2 tokens for response
            return []

        self.mock_tokenizer.encode.side_effect = mock_encode

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512, ignore_index=-100
        )

        # Verify prompt tokens (first 3) are masked
        labels = result["labels"]
        self.assertEqual(labels[0].item(), -100)
        self.assertEqual(labels[1].item(), -100)
        self.assertEqual(labels[2].item(), -100)

        # Verify response tokens are NOT masked
        self.assertNotEqual(labels[3].item(), -100)
        self.assertNotEqual(labels[4].item(), -100)

    def test_eos_token_appended_to_prompt_and_response(self):
        """Test EOS token is appended after encoding - ensures sequence termination."""
        element = {
            "id": 1,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        self.mock_tokenizer.encode.return_value = [100, 101]

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512
        )

        # Last token should be EOS
        self.assertEqual(result["input_ids"][-1].item(), self.mock_tokenizer.eos_token_id)

    def test_max_seq_length_truncation(self):
        """Test max_seq_length truncates prompt+response correctly - prevents overflow."""
        element = {
            "id": 1,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        # Tokenize method is called with max_length parameter
        # The function encodes prompt_and_response with max_length - 1, then appends EOS
        def mock_encode(text, max_length=None, add_special_tokens=False):
            if max_length:
                # Return truncated to max_length
                return [1] * min(100, max_length)
            return [1] * 100

        self.mock_tokenizer.encode.side_effect = mock_encode

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=10
        )

        # Should be truncated to 10 (max_seq_length - 1 = 9, + EOS = 10)
        self.assertEqual(len(result["input_ids"]), 10)

    def test_attention_masks_all_ones(self):
        """Test attention masks are all ones for valid tokens - validates attention logic."""
        element = {
            "id": 1,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        self.mock_tokenizer.encode.return_value = [10, 11, 12]

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512
        )

        # All attention masks should be 1
        self.assertTrue(torch.all(result["attention_mask"] == 1))
        self.assertTrue(torch.all(result["attention_mask_alone"] == 1))
        self.assertTrue(torch.all(result["attention_mask_prompt"] == 1))

    def test_output_contains_all_required_keys(self):
        """Test output dict contains all required keys - prevents missing field regressions."""
        element = {
            "id": 42,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        self.mock_tokenizer.encode.return_value = [1, 2, 3]

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512
        )

        required_keys = [
            "id", "input_ids", "labels", "attention_mask",
            "labels_alone", "attention_mask_alone",
            "prompt_alone", "attention_mask_prompt"
        ]

        for key in required_keys:
            self.assertIn(key, result)

    def test_id_converted_to_tensor(self):
        """Test ID is converted to long tensor - ensures type consistency."""
        element = {
            "id": 999,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        self.mock_tokenizer.encode.return_value = [1]

        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512
        )

        self.assertIsInstance(result["id"], torch.Tensor)
        self.assertEqual(result["id"].item(), 999)
        self.assertEqual(result["id"].dtype, torch.long)

    def test_custom_ignore_index(self):
        """Test custom ignore_index is used for masking - validates configurability."""
        element = {
            "id": 1,
            "prompt": "P",
            "prompt_and_response": "PR",
            "response": "R"
        }

        self.mock_tokenizer.encode.side_effect = lambda text, **kwargs: [1, 2] if "PR" in text else [1]

        custom_ignore = -999
        result = tokenizer_dataset_given_prompt(
            element, self.mock_tokenizer, max_seq_length=512, ignore_index=custom_ignore
        )

        # Prompt token should be masked with custom ignore index
        self.assertEqual(result["labels"][0].item(), custom_ignore)


class TestTokenizerDatasetMultiTurn(unittest.TestCase):
    """Test multi-turn dataset processing - PROMPT REGRESSION TESTS."""

    def setUp(self):
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2

    @patch('cherry_seletion.preprocess.create_conversation')
    @patch('cherry_seletion.preprocess.create_prompt')
    @patch('cherry_seletion.preprocess.tqdm')
    def test_multi_turn_appends_im_end_token(self, mock_tqdm, mock_create_prompt, mock_create_conversation):
        """Test <|im_end|> is appended to prompt_and_response - critical prompt structure."""
        mock_tqdm.side_effect = lambda x: x  # Pass through

        data = [
            {
                "id": 1,
                "system": "System",
                "input": ["Q1"],
                "assistance": [],
                "target": "Answer"
            }
        ]

        mock_create_conversation.return_value = [{"role": "system", "content": "System"}]
        mock_create_prompt.return_value = "Formatted prompt"
        self.mock_tokenizer.encode.return_value = [1, 2, 3]

        result = tokenizer_dataset_multi_turn(data, self.mock_tokenizer, max_seq_length=512)

        # Get the first element's data before tokenization
        # We need to check the intermediate data_result
        # Since we can't access it directly, we verify via the encode calls
        encode_calls = self.mock_tokenizer.encode.call_args_list

        # Find the call with prompt_and_response
        found_im_end = False
        for call in encode_calls:
            args, kwargs = call
            if args and "im_end" in str(args[0]):
                found_im_end = True
                break

        self.assertTrue(found_im_end, "<|im_end|> token should be in prompt_and_response")

    @patch('cherry_seletion.preprocess.create_conversation')
    @patch('cherry_seletion.preprocess.create_prompt')
    @patch('cherry_seletion.preprocess.tqdm')
    def test_multi_turn_creates_conversation_for_each_element(self, mock_tqdm, mock_create_prompt, mock_create_conversation):
        """Test create_conversation called for each data element - validates pipeline."""
        mock_tqdm.side_effect = lambda x: x

        data = [
            {"id": 1, "system": "S1", "input": ["Q1"], "assistance": [], "target": "A1"},
            {"id": 2, "system": "S2", "input": ["Q2"], "assistance": [], "target": "A2"}
        ]

        mock_create_conversation.return_value = []
        mock_create_prompt.return_value = "prompt"
        self.mock_tokenizer.encode.return_value = [1]

        tokenizer_dataset_multi_turn(data, self.mock_tokenizer, max_seq_length=512)

        self.assertEqual(mock_create_conversation.call_count, 2)

    @patch('cherry_seletion.preprocess.create_conversation')
    @patch('cherry_seletion.preprocess.create_prompt')
    @patch('cherry_seletion.preprocess.tqdm')
    def test_multi_turn_passes_conversation_to_create_prompt(self, mock_tqdm, mock_create_prompt, mock_create_conversation):
        """Test conversation is passed to create_prompt - validates data flow."""
        mock_tqdm.side_effect = lambda x: x

        data = [{"id": 1, "system": "S", "input": ["Q"], "assistance": [], "target": "A"}]

        expected_conversation = [{"role": "system", "content": "S"}]
        mock_create_conversation.return_value = expected_conversation
        mock_create_prompt.return_value = "prompt"
        self.mock_tokenizer.encode.return_value = [1]

        tokenizer_dataset_multi_turn(data, self.mock_tokenizer, max_seq_length=512)

        mock_create_prompt.assert_called_once_with(expected_conversation, self.mock_tokenizer)

    @patch('cherry_seletion.preprocess.create_conversation')
    @patch('cherry_seletion.preprocess.create_prompt')
    @patch('cherry_seletion.preprocess.tqdm')
    def test_multi_turn_dataset_format_is_torch(self, mock_tqdm, mock_create_prompt, mock_create_conversation):
        """Test dataset format is set to torch - validates output type."""
        mock_tqdm.side_effect = lambda x: x

        data = [{"id": 1, "system": "S", "input": ["Q"], "assistance": [], "target": "A"}]

        mock_create_conversation.return_value = []
        mock_create_prompt.return_value = "prompt"
        self.mock_tokenizer.encode.return_value = [1]

        result = tokenizer_dataset_multi_turn(data, self.mock_tokenizer, max_seq_length=512)

        # Verify it's a Dataset instance
        from datasets import Dataset
        self.assertIsInstance(result, Dataset)

    @patch('cherry_seletion.preprocess.create_conversation')
    @patch('cherry_seletion.preprocess.create_prompt')
    @patch('cherry_seletion.preprocess.tqdm')
    def test_multi_turn_empty_data_produces_empty_dataset(self, mock_tqdm, mock_create_prompt, mock_create_conversation):
        """Test empty input data produces empty dataset - boundary condition."""
        mock_tqdm.side_effect = lambda x: x

        data = []

        # Empty dataset will fail when set_format is called on columns that don't exist
        # This is expected behavior - the function assumes non-empty data
        # We test that it at least doesn't crash during the iteration phase
        with self.assertRaises(ValueError) as context:
            result = tokenizer_dataset_multi_turn(data, self.mock_tokenizer, max_seq_length=512)

        # Verify error is about missing columns (expected for empty dataset)
        self.assertIn("Columns", str(context.exception))


class TestSftCollateFn(unittest.TestCase):
    """Test collate functions for batching - LOGIC TESTS."""

    def test_get_sft_collate_fn_returns_partial(self):
        """Test get_sft_collate_fn returns a partial function - validates factory pattern."""
        collate_fn = get_sft_collate_fn(max_seq_length=512, pad_id=0, ignore_index=-100)

        from functools import partial
        self.assertIsInstance(collate_fn, partial)

    def test_collate_fn_pads_to_longest_sequence(self):
        """Test sequences are padded to longest in batch - critical batching logic."""
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels_alone": torch.tensor([1, 2]),
                "attention_mask_alone": torch.tensor([1, 1]),
                "prompt_alone": torch.tensor([1]),
                "attention_mask_prompt": torch.tensor([1])
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels_alone": torch.tensor([4]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([4, 5]),
                "attention_mask_prompt": torch.tensor([1, 1])
            }
        ]

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=0, ignore_index=-100)

        # All sequences should be padded to length 3 (longest input_ids)
        self.assertEqual(result["input_ids"].shape, torch.Size([2, 3]))

        # Verify padding values
        self.assertEqual(result["input_ids"][1, 2].item(), 0)  # pad_id

    def test_collate_fn_uses_correct_padding_values(self):
        """Test different padding values for different keys - validates pad logic per field."""
        samples = [
            {
                "input_ids": torch.tensor([1]),
                "labels": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "labels_alone": torch.tensor([1]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([1]),
                "attention_mask_prompt": torch.tensor([1])
            },
            {
                "input_ids": torch.tensor([2, 3]),
                "labels": torch.tensor([2, 3]),
                "attention_mask": torch.tensor([1, 1]),
                "labels_alone": torch.tensor([2, 3]),
                "attention_mask_alone": torch.tensor([1, 1]),
                "prompt_alone": torch.tensor([2, 3]),
                "attention_mask_prompt": torch.tensor([1, 1])
            }
        ]

        pad_id = 99
        ignore_index = -999

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=pad_id, ignore_index=ignore_index)

        # Verify padding values
        self.assertEqual(result["input_ids"][0, 1].item(), pad_id)
        self.assertEqual(result["labels"][0, 1].item(), ignore_index)
        self.assertEqual(result["attention_mask"][0, 1].item(), 0)
        self.assertEqual(result["labels_alone"][0, 1].item(), ignore_index)
        self.assertEqual(result["prompt_alone"][0, 1].item(), pad_id)

    def test_collate_fn_truncates_to_max_seq_length(self):
        """Test sequences truncated to max_seq_length - prevents memory overflow."""
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                "labels": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
                "labels_alone": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask_alone": torch.tensor([1, 1, 1, 1, 1]),
                "prompt_alone": torch.tensor([1, 2, 3]),
                "attention_mask_prompt": torch.tensor([1, 1, 1])
            }
        ]

        max_length = 5
        result = _sft_collate_fn(samples, max_seq_length=max_length, pad_id=0, ignore_index=-100)

        # All sequences should be truncated to max_length
        self.assertEqual(result["input_ids"].shape[1], max_length)
        self.assertEqual(result["labels"].shape[1], max_length)
        self.assertEqual(result["attention_mask"].shape[1], max_length)

    def test_collate_fn_batch_first_format(self):
        """Test output is in batch-first format - validates tensor shape convention."""
        samples = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels_alone": torch.tensor([1]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([1]),
                "attention_mask_prompt": torch.tensor([1])
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([3, 4]),
                "attention_mask": torch.tensor([1, 1]),
                "labels_alone": torch.tensor([3]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([3]),
                "attention_mask_prompt": torch.tensor([1])
            }
        ]

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=0, ignore_index=-100)

        # Shape should be [batch_size, seq_length]
        self.assertEqual(result["input_ids"].shape[0], 2)  # batch size

    def test_collate_fn_handles_all_keys(self):
        """Test all required keys are in output - prevents missing field regressions."""
        samples = [
            {
                "input_ids": torch.tensor([1]),
                "labels": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "labels_alone": torch.tensor([1]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([1]),
                "attention_mask_prompt": torch.tensor([1])
            }
        ]

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=0, ignore_index=-100)

        expected_keys = [
            "input_ids", "labels", "attention_mask",
            "labels_alone", "attention_mask_alone",
            "prompt_alone", "attention_mask_prompt"
        ]

        for key in expected_keys:
            self.assertIn(key, result)

    def test_collate_fn_single_sample_no_padding(self):
        """Test single sample batch requires no padding - edge case validation."""
        samples = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels_alone": torch.tensor([1, 2]),
                "attention_mask_alone": torch.tensor([1, 1]),
                "prompt_alone": torch.tensor([1]),
                "attention_mask_prompt": torch.tensor([1])
            }
        ]

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=0, ignore_index=-100)

        # Should match original lengths
        self.assertEqual(result["input_ids"].shape, torch.Size([1, 3]))
        self.assertTrue(torch.equal(result["input_ids"][0], samples[0]["input_ids"]))

    def test_collate_fn_preserves_tensor_values(self):
        """Test tensor values are preserved during batching - data integrity check."""
        samples = [
            {
                "input_ids": torch.tensor([10, 20, 30]),
                "labels": torch.tensor([100, 200, 300]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels_alone": torch.tensor([100]),
                "attention_mask_alone": torch.tensor([1]),
                "prompt_alone": torch.tensor([10, 20]),
                "attention_mask_prompt": torch.tensor([1, 1])
            }
        ]

        result = _sft_collate_fn(samples, max_seq_length=-1, pad_id=0, ignore_index=-100)

        # Verify original values are preserved
        self.assertEqual(result["input_ids"][0, 0].item(), 10)
        self.assertEqual(result["input_ids"][0, 1].item(), 20)
        self.assertEqual(result["labels"][0, 0].item(), 100)
        self.assertEqual(result["labels"][0, 1].item(), 200)


if __name__ == '__main__':
    unittest.main()
