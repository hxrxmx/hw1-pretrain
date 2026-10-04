import your_solution as solution


def test_padding_is_ignored_in_loss():
    features = [
        {
            'input_ids': [10, 11, 2],
            'attention_mask': [1, 1, 0],
            'labels': [10, 11, 2],
        }
    ]
    batch = solution.causal_lm_collator(features)
    assert batch['labels'].tolist() == [[10, 11, -100]]


if __name__ == '__main__':
    test_padding_is_ignored_in_loss()
    print('padding mask ok')
