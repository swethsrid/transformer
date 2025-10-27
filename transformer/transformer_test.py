from transformer_hw import *
import unittest

class TestTransformerFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up reference model and move it to device once for all tests"""
        global reference_gpt2, tokens, cache
        # Move reference model to device
        reference_gpt2 = reference_gpt2.to(device)
        # Regenerate tokens and cache with model on device
        reference_text = "Today we are going to implement a Transformer from scratch!"
        tokens = reference_gpt2.to_tokens(reference_text).to(device)
        logits, cache = reference_gpt2.run_with_cache(tokens)
    
    def rand_float_test(self, cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = torch.randn(shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape)
        # Don't assert shape equality - different layers transform shapes differently

    def rand_int_test(self, cls, shape):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        random_input = torch.randint(100, 1000, shape).to(device)
        print("Input shape:", random_input.shape)
        output = layer(random_input)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape)
        # Don't assert shape equality - embedding layers change dimensions

    def load_gpt2_test(self, cls, gpt2_layer, input_tensor):
        cfg = Config(debug=True)
        layer = cls(cfg).to(device)
        layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
        
        # Ensure input is on device
        input_tensor = input_tensor.to(device)
        
        print("Input shape:", input_tensor.shape)
        output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        print("Output shape:", output.shape)
        
        # Get reference output - handle Attention layer specially
        if cls == Attention:
            # Attention layers need query, key, and value inputs
            reference_output = gpt2_layer(input_tensor, input_tensor, input_tensor)
        else:
            reference_output = gpt2_layer(input_tensor)
        
        print("Reference output shape:", reference_output.shape, "\n")
        
        comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
        percentage_correct = comparison.sum().item() / comparison.numel()
        print(f"{percentage_correct:.2%} of the values are correct")
        
        self.assertTrue(comparison.all(),
            f"Only {percentage_correct:.2%} of the values are correct\n")

    def testLayerNorm(self):
        print("\n=== Testing LayerNorm ===")
        self.rand_float_test(LayerNorm, [2, 4, 768])
        self.load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])

    def testEmbed(self):
        print("\n=== Testing Embed ===")
        self.rand_int_test(Embed, [2, 4])
        self.load_gpt2_test(Embed, reference_gpt2.embed, tokens)

    def testPosEmbed(self):
        print("\n=== Testing PosEmbed ===")
        self.rand_int_test(PosEmbed, [2, 4])
        self.load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

    def testAttention(self):
        print("\n=== Testing Attention ===")
        self.rand_float_test(Attention, [2, 4, 768])
        self.load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

    def testMLP(self):
        print("\n=== Testing MLP ===")
        self.rand_float_test(MLP, [2, 4, 768])
        self.load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

    def testTransformer(self):
        print("\n=== Testing TransformerBlock ===")
        self.rand_float_test(TransformerBlock, [2, 4, 768])
        self.load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

    def testUnembed(self):
        print("\n=== Testing Unembed ===")
        self.rand_float_test(Unembed, [2, 4, 768])
        self.load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

    def testDemo(self):
        print("\n=== Testing DemoTransformer ===")
        self.rand_int_test(DemoTransformer, [2, 4])
        self.load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

    def testGeneration(self):
        print("\n=== Testing Generation ===")
        demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
        demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
        
        start_sequence = "Today I was walking home, when suddenly"
        start_tokens = reference_gpt2.to_tokens(start_sequence, prepend_bos=True)
        max_new_tokens = 20
        
        # Generate tokens using greedy decoding
        generated_tokens = greedy_decode(demo_gpt2, start_tokens, max_new_tokens)
        
        # Decode generated tokens back to text (skip BOS token)
        generated_text = reference_gpt2.to_string(generated_tokens[0, 1:])
        print("Generated Text:", generated_text)
        
        reference_generation = reference_gpt2.generate(start_sequence,
            max_new_tokens=max_new_tokens,
            stop_at_eos=False,
            do_sample=False)
        
        print("Reference Text:", reference_generation)
        self.assertEqual(reference_generation, generated_text)

if __name__ == '__main__':
    unittest.main()
