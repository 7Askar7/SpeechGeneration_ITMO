import torch
import torchaudio
import Levenshtein
import pandas as pd
from wav2vec2decoder import Wav2Vec2Decoder

# Define parameter combinations to test
beam_widths = [2, 3, 5, 10]
alphas = [0.5, 1.0, 1.5, 2.0]
betas = [0.5, 1.0, 1.5, 2.0]

# Test samples to use - we'll use a subset of the original test set for efficiency
test_samples = [
    ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
    ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
    ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
]

# Load audio data once to avoid reloading for each test
audio_data = []
for audio_path, transcription in test_samples:
    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"
    audio_data.append((audio_input, transcription))

# Results collection
results = []

# Test each parameter combination
for beam_width in beam_widths:
    for alpha in alphas:
        for beta in betas:
            print(f"Testing: beam_width={beam_width}, alpha={alpha}, beta={beta}")
            
            # Initialize decoder with current parameters
            decoder = Wav2Vec2Decoder(
                beam_width=beam_width,
                alpha=alpha,
                beta=beta
            )
            
            # Test each decoding method for each audio sample
            for (audio_input, true_transcription), (audio_path, _) in zip(audio_data, test_samples):
                for method in ["beam", "beam_lm", "beam_lm_rescore"]:
                    transcript = decoder.decode(audio_input, method=method)
                    
                    # Calculate Levenshtein distance
                    distance = Levenshtein.distance(true_transcription, transcript.strip())
                    
                    # Add result to collection
                    results.append({
                        "beam_width": beam_width,
                        "alpha": alpha,
                        "beta": beta,
                        "method": method,
                        "audio": audio_path.split('/')[-1],
                        "levenshtein_distance": distance
                    })
            
            # Create a checkpoint to save results after each parameter combination
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_csv(f"parameter_test_checkpoint.csv", index=False)

# Process and analyze results
results_df = pd.DataFrame(results)
results_df.to_csv("parameter_test_results.csv", index=False)

# Analyze results for each method
print("\nSummary of Best Parameter Combinations by Method:")
for method in ["beam", "beam_lm", "beam_lm_rescore"]:
    method_results = results_df[results_df["method"] == method]
    
    # Get average Levenshtein distance for each parameter combination
    avg_results = method_results.groupby(["beam_width", "alpha", "beta"])["levenshtein_distance"].mean().reset_index()
    
    # Find best parameter combination
    best_params = avg_results.loc[avg_results["levenshtein_distance"].idxmin()]
    
    print(f"\n{method} decoding:")
    print(f"  Best parameters: beam_width={best_params['beam_width']}, alpha={best_params['alpha']}, beta={best_params['beta']}")
    print(f"  Average Levenshtein distance: {best_params['levenshtein_distance']:.2f}")

print("\nComplete results saved to parameter_test_results.csv") 