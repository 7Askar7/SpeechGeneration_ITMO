from typing import List, Tuple
import heapq
import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/4-gram.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            device=None
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
            device (torch.device): Device to use for computation
        """
        # Автоматическое определение устройства, если не указано явно
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Загрузка моделей
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        best_log_prob = torch.argmax(log_probs, dim=-1)

        pred_tokens = []
        prev_token = None

        for token_idx in best_log_prob:
            token_idx = token_idx.cpu().item()

            if token_idx == self.blank_token_id:
                prev_token = self.blank_token_id
                continue

            if token_idx == prev_token:
                continue

            pred_tokens.append(token_idx)
            prev_token = token_idx
        
        transcript = "".join([self.vocab[token_idx] for token_idx in pred_tokens])
        transcript = transcript.replace(self.word_delimiter, " ")
        
        return transcript

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        beam = [(0.0, [], None)]
        
        for t in range(log_probs.size(0)):
            topk_probs, topk_indices = torch.topk(log_probs[t], k=min(5, log_probs.size(1)))
            topk_probs = topk_probs.cpu().tolist()
            topk_indices = topk_indices.cpu().tolist()
            
            new_beam = []
            
            for log_p, prefix, prev_token in beam:
                for i, v in enumerate(topk_indices):
                    if v == prev_token and v != self.blank_token_id:
                        continue
                    
                    new_log_p = log_p + topk_probs[i]
                    
                    if v == self.blank_token_id:
                        new_beam.append((new_log_p, prefix.copy(), prev_token))
                    else:
                        new_prefix = prefix.copy()
                        new_prefix.append(v)
                        new_beam.append((new_log_p, new_prefix, v))

            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:self.beam_width]
        
        final_beams = []
        for log_p, prefix, _ in beam:
            tokens = [self.vocab[token_idx] for token_idx in prefix]
            
            words = []
            current_word = ""
            
            for i, token in enumerate(tokens):

                if token == self.word_delimiter:
                    if current_word:
                        words.append(current_word)
                        current_word = ""
                else:
                    if (i > 0 and tokens[i-1] == self.word_delimiter and 
                        i < len(tokens)-1 and tokens[i+1] == self.word_delimiter and
                        token in "AEIOUTSHD"):
                        continue
                    
                    current_word += token
            
            if current_word:
                words.append(current_word)
                
            transcript = " ".join(words)
            final_beams.append((log_p, prefix, transcript))
        
        final_beams = sorted(final_beams, key=lambda x: x[0], reverse=True)
        
        if return_beams:
            return [(log_p, prefix) for log_p, prefix, _ in final_beams]
        else:
            return final_beams[0][2] if final_beams else ""

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        greedy_transcript = self.greedy_decode(logits)
        greedy_words = greedy_transcript.split()
        
        beams = self.beam_search_decode(logits, return_beams=True)
        
        all_candidates = []
        
        greedy_score = 0.0 
        for word in greedy_words:
            word_score = self.lm_model.score(word, bos=True, eos=False)
            greedy_score += word_score * 0.1  
        
        all_candidates.append((greedy_score + 5.0, [], greedy_transcript))
        
        for acoustic_score, prefix in beams:
            tokens = [self.vocab[token_idx] for token_idx in prefix]
            words = []
            current_word = ""
            
            for i, token in enumerate(tokens):
                if token == self.word_delimiter:
                    if current_word and len(current_word) > 1:
                        if not any(c*3 in current_word for c in "AEIOUTSHD"):
                            words.append(current_word)
                    current_word = ""
                else:
                    if (i > 0 and tokens[i-1] == self.word_delimiter and 
                        i < len(tokens)-1 and tokens[i+1] == self.word_delimiter and
                        token in "AEIOUTSHD"):
                        continue
                    current_word += token
            
            if current_word and len(current_word) > 1:
                if not any(c*3 in current_word for c in "AEIOUTSHD"):
                    words.append(current_word)
            
            transcript = " ".join(words)

            if not transcript or len(transcript) < 5:
                continue

            import Levenshtein
            if len(greedy_transcript) > 0:
                similarity = 1.0 - Levenshtein.distance(transcript, greedy_transcript) / max(len(transcript), len(greedy_transcript))
                similarity_bonus = 2.0 * similarity if similarity > 0.6 else 0.0
            else:
                similarity_bonus = 0.0
            
            lm_score = self.lm_model.score(transcript, bos=True, eos=True)
            
            total_score = (
                acoustic_score * 0.3 +  
                self.alpha * lm_score * 1.5 + 
                self.beta * len(words) +  
                0.01 * sum(len(w) for w in words) +  
                similarity_bonus 
            )
            
            all_candidates.append((total_score, prefix, transcript))
        
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        
        return all_candidates[0][2] if all_candidates else greedy_transcript

    def lm_rescore(self, beams: List[Tuple[float, List[int]]], max_candidates: int = 10) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (log_prob, prefix)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        candidates = beams[:max_candidates]
        best_score = float('-inf')
        best_transcript = ""
        
        acoustic_scores = [log_prob for log_prob, _ in candidates]
        min_score = min(acoustic_scores)
        max_score = max(acoustic_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        

        def is_unlikely_sequence(word):
            unlikely_patterns = ['AAA', 'EEE', 'III', 'OOO', 'UUU', 'TTT', 'SSS', 'HHH']
            return any(pattern in word for pattern in unlikely_patterns)
        
        for log_prob, prefix in candidates:
            tokens = [self.vocab[token_idx] for token_idx in prefix]
            words = []
            current_word = ""
            
            for i, token in enumerate(tokens):
                if token == self.word_delimiter:
                    if current_word and len(current_word) > 1 and not is_unlikely_sequence(current_word):
                        words.append(current_word)
                    current_word = ""
                else:
                    if (i > 0 and tokens[i-1] == self.word_delimiter and 
                        i < len(tokens)-1 and tokens[i+1] == self.word_delimiter and
                        token in "AEIOUTSHD"):
                        continue
                    current_word += token

            if current_word and len(current_word) > 1 and not is_unlikely_sequence(current_word):
                words.append(current_word)

            transcript = " ".join(words)
            
            if not transcript or len(transcript) < 5:
                continue

            norm_acoustic_score = (log_prob - min_score) / score_range

            boosted_transcript = transcript
            lm_score = self.lm_model.score(boosted_transcript, bos=True, eos=True)
            
            acoustic_weight = 0.4  
            lm_weight = self.alpha * 2.0  
            
            word_count = len(words)
            char_count = sum(len(word) for word in words)
            word_bonus = self.beta * word_count  
            char_bonus = 0.01 * char_count 
  
            short_word_count = sum(1 for word in words if len(word) == 1)
            short_word_penalty = -0.2 * short_word_count if short_word_count > 2 else 0
            
            final_score = (acoustic_weight * norm_acoustic_score + 
                          lm_weight * lm_score + 
                          word_bonus + 
                          char_bonus +
                          short_word_penalty)
            
            if final_score > best_score:
                best_score = final_score
                best_transcript = transcript
        
        return best_transcript

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        audio_input = audio_input.to(self.device)
        
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(inputs['input_values'].squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding") 
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
