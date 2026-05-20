# midichords Python module

r'''###############################################################################
###################################################################################
#
#
#	midichords Python module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Critical dependencies
#
#   !pip install huggingface_hub
#   !pip install hf-transfer
#   !pip install ipywidgets
#   !pip install tqdm
#
#   !pip install torch
#   !pip install einops
@   !pip install einx
#   !pip install torch-summary
#   !pip install matplotlib
#   !pip install numpy==1.26.4
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading midichords Python module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midichords module version', __version__)
print('=' * 70)

###################################################################################
###################################################################################

import os, copy, math, shutil

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import List, Optional, Union, Tuple, Dict, Any

from functools import lru_cache

import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from .x_transformer_2_3_1 import TransformerWrapper, Encoder

from torchsummary import summary

from . import TMIDIX

from huggingface_hub import hf_hub_download, snapshot_download

print('=' * 70)
print('PyTorch version:', torch.__version__)
print('=' * 70)

###################################################################################


###################################################################################

def download_model(repo_id: str = 'projectlosangeles/midichords',
                   filename: str = 'midichords_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth',
                   local_dir: str = './midichords-models/',
                   verbose: bool = True,
                   **kwargs: dict[str, Any]
                  ) -> str:
    
    """
    Helper function that downloads pre-trained midichords models from Hugging Face
    
    Returns
    -------
    Downloaded model checkpoint file path string
    """
    
    if verbose:
        print('=' * 70)
        print('Downloading model...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='model',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    if verbose:    
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def load_model(model_path: str = './midichords-models/midichords_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth',
               dim: int = 512,
               depth: int = 16,
               heads: int = 16,
               max_seq_len: int = 2048,
               vocab_size=335,
               dtype: torch.dtype = torch.bfloat16,
               device: str = 'cuda',
               verbose: bool = True
              ) -> str:

    if verbose:
        print('=' * 70)
        print('midichords model loader')
        print('=' * 70)
        print('Initializing model...')

    ctx = torch.amp.autocast(device_type=device, dtype=dtype)

    model = TransformerWrapper(
                num_tokens=vocab_size,
                max_seq_len=max_seq_len,
                attn_layers=Encoder(
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    rotary_pos_emb=True,
                    attn_flash=True,
                ),
    )

    if verbose:
        print('=' * 70)
        print('Loading model checkpoint...')
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    if verbose:
        print('=' * 70)
    
    model.to(device)
    model.eval()

    if verbose:
        print('Done!')

        print('=' * 70)
        print('Model Summary')
        summary(model)

    return model, ctx, dtype

###################################################################################

def load_embeddings(embeddings_path: str = './midichords-embeddings/discover_midi_dataset_37292_genres_midis_embeddings_cc_by_nc_sa.npy',
                    midi_names_key: str = 'midi_names',
                    midi_embeddings_key: str = 'midi_embeddings',
                    verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Helper function that loads pre-computed embeddings file

    Returns
    -------
    Tuple of nd.arrays (midi_names_arr, midi_embeddings_arr)
    """

    if verbose:
        print('=' * 70)
        print('Loading embeddings...')
        
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return embeddings_data[midi_names_key], embeddings_data[midi_embeddings_key]

###################################################################################

def save_embeddings(embeddings_name_strings: list[str],
                    embeddings: Union[torch.Tensor, np.ndarray],
                    name_strings_key: str = 'midi_names',
                    embeddings_key: str = 'midi_embeddings',
                    output_file_name: str = 'saved_midi_embeddings.npy',
                    return_merged_array: bool = False,
                    verbose=True
                   ) -> Union[np.ndarray, None]:

    """Save a list of name strings and their corresponding embedding vectors into a NumPy structured array
    and optionally persist it to disk.
    
    This function builds a NumPy structured array with two fields (one for the name strings and one for
    the embedding vectors), populates it from the provided inputs, casts embeddings to `np.float32`,
    and either returns the merged structured array or saves it to disk using `np.save`.
    
    Parameters
    ----------
    embeddings_name_strings : list[str]
        Sequence of Python strings that identify each embedding (e.g., filenames, IDs, labels).
        The length of this list determines the number of rows in the resulting structured array.
    embeddings : Union[torch.Tensor, np.ndarray]
        2D array-like of shape `(n, D)` containing the embedding vectors, where `n` must equal
        `len(embeddings_name_strings)` and `D` is the embedding dimensionality. If a `torch.Tensor`
        is provided it will be converted to a NumPy array with `.numpy()` (no automatic `.cpu()`
        or `.detach()` is performed by this function).
    name_strings_key : str, optional
        Field name to use for the name strings in the structured dtype (default: `'midi_names'`).
    embeddings_key : str, optional
        Field name to use for the embedding vectors in the structured dtype (default:
        `'midi_embeddings'`).
    output_file_name : str, optional
        Path (filename) where the structured array will be saved with `np.save` if
        `return_merged_array` is `False` (default: `'saved_midi_embeddings.npy'`).
    return_merged_array : bool, optional
        If `True`, the function returns the constructed structured NumPy array and does not write
        anything to disk. If `False`, the array is saved to `output_file_name` and the function
        returns `None` (default: `False`).
    verbose : bool, optional
        If `True`, print progress and diagnostic messages to stdout (default: `True`).
    
    Returns
    -------
    Union[np.ndarray, None]
        - If `return_merged_array` is `True`: the NumPy structured array of length `n` with dtype
          `[(name_strings_key, object), (embeddings_key, np.float32, (D,))]`.
        - If `return_merged_array` is `False`: `None` (the array is saved to `output_file_name`).
    
    Raises
    ------
    ValueError
        - If `embeddings` does not have a second dimension (i.e., is not 2D) so the embedding
          dimensionality `D` cannot be determined.
        - If the number of rows in `embeddings` does not match `len(embeddings_name_strings)`.
    TypeError
        - If `embeddings_name_strings` is not a sequence with a definable length.
    Exception
        - Any unexpected exceptions raised while reading attributes (e.g., `.shape`, `.dtype`) or
          during `np.save` will propagate to the caller.
    
    Notes
    -----
    - The function constructs a structured dtype where the name field uses Python `object` to allow
      variable-length strings and the embedding field is a fixed-size `np.float32` vector of length `D`.
    - Embeddings are explicitly cast to `np.float32` before assignment; this may change precision.
    - When a `torch.Tensor` is passed, the function calls `.numpy()` directly. If the tensor is on a
      GPU or requires gradient, callers should ensure it is detached and moved to CPU first (e.g.,
      `embeddings.detach().cpu()`), otherwise `.numpy()` may raise an error.
    - The file is written using `np.save`, producing a `.npy` file that can be loaded with `np.load`.
    - Verbose logging is intended for debugging and progress visibility; set `verbose=False` to silence.
    
    Example
    -------
    >>> # embeddings as numpy array
    >>> names = ['song_a.mid', 'song_b.mid']
    >>> embs = np.random.randn(2, 512)
    >>> save_embeddings(names, embs, output_file_name='embs.npy', verbose=False)
    >>> # embeddings as torch tensor, return array instead of saving
    >>> import torch
    >>> t = torch.randn(2, 512)
    >>> arr = save_embeddings(names, t, return_merged_array=True, verbose=False)
    >>> assert arr.dtype == np.dtype([('midi_names', object), ('midi_embeddings', np.float32, (512,))])
    
    """

    if verbose:
        print('=' * 70)
        print('Saving embeddings...')
        print('=' * 70)
        print("[save_embeddings]: called with parameters:")
        print(f"  number of name strings provided: {len(embeddings_name_strings)}")
        print(f"  output_file_name: {output_file_name}")
        print(f"  name_strings_key: {name_strings_key}")
        print(f"  embeddings_key: {embeddings_key}")
        print(f"  return_merged_array: {return_merged_array}")
        print(f"  verbose: {verbose}")
        print('=' * 70)

    # Convert torch tensor to numpy if needed
    if type(embeddings) == torch.Tensor:
        if verbose:
            print("[save_embeddings]: embeddings is a torch.Tensor, converting to numpy array with .numpy()")
        embeddings = embeddings.cpu().numpy()
    elif type(embeddings) == list:
        if verbose:
                print("[save_embeddings]: embeddings is a list, converting to numpy array")
        embeddings = np.array(embeddings)
    else:
        if verbose:
            print(f"[save_embeddings]: embeddings is of type {type(embeddings)}; no conversion performed")

    # Basic shape and length checks
    try:
        n = len(embeddings_name_strings)
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR computing length of embeddings_name_strings:", e)
        raise

    try:
        D = embeddings.shape[1]
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR reading embeddings.shape[1]:", e)
            print("  embeddings.shape is:", getattr(embeddings, "shape", None))
        raise

    if verbose:
        print(f"[save_embeddings]: determined n = {n} (number of entries)")
        print(f"[save_embeddings]: determined D = {D} (embedding dimensionality)")
        print("[save_embeddings]: preparing numpy structured dtype for storage")

    dtype = np.dtype([
        (name_strings_key, object),              # variable-length Python strings
        (embeddings_key, embeddings.dtype, (D,))       # fixed-size embedding vector
    ])

    if verbose:
        print("[save_embeddings]: dtype constructed as:")
        print(f"  {dtype}")

    # Create empty structured array
    if verbose:
        print(f"[save_embeddings]: allocating empty numpy array of length {n} with dtype above")
    arr = np.empty(n, dtype=dtype)

    # Fill name strings
    if verbose:
        print("[save_embeddings]: assigning name strings to structured array")
        print(f"  first 5 name strings (or fewer): {embeddings_name_strings[:5]}")
    arr[name_strings_key] = embeddings_name_strings

    # Cast embeddings to float32 and assign
    if verbose:
        print("[save_embeddings]: assigning embeddings to structured array")
        print(f"  embeddings original dtype: {getattr(embeddings, 'dtype', 'unknown')}")
        print(f"  embeddings shape: {getattr(embeddings, 'shape', 'unknown')}")
    arr[embeddings_key] = embeddings

    if return_merged_array:
        if verbose:
            print('=' * 70)
            print("[save_embeddings]: return_merged_array is True; returning the merged structured array without saving to disk")
            print(f"  returning array with length {len(arr)} and dtype {arr.dtype}")
            print('=' * 70)
            print('Done!')
            print('=' * 70)
        return arr

    # Save to disk
    if verbose:
        print('=' * 70)
        print(f"[save_embeddings]: return_merged_array is False; saving structured array to '{output_file_name}' using np.save")
    np.save(output_file_name, arr)
    if verbose:
        print(f"[save_embeddings]: save complete. File written: {output_file_name}")
        print(f"  saved array length: {len(arr)}; dtype: {arr.dtype}")
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
###################################################################################

def midi_to_tokens(midi_file_path: str,
                   max_seq_len: int = 2048,
                   transpose_factor: int = 6,
                   verbose: bool = True
                  )-> list[list[int]]:
    
    """
    Convert a single-track MIDI file into one or more compact token sequences suitable for model input.

    This function performs a sequence of TMIDIX preprocessing steps to extract an
    "enhanced score" from a MIDI file, normalizes and clips timing/pitch values,
    optionally generates transposed variants, and encodes events into a compact
    integer token stream.

    Key processing stages
    - Load MIDI and convert to a single-track millisecond score.
    - Produce an enhanced-score with sustain applied.
    - Extract solo-piano notes and recalculate/augment timings.
    - Remove duplicate pitches and fix note durations.
    - Convert to a delta-style event list and clip timing values to 0..127.
    - For each transpose variant, build a token sequence where:
        * nonzero delta times are appended as-is (0..127),
        * note-on events are encoded as two tokens: (duration + 128) and (pitch + 256).
      The initial token of each sequence is 0 (start token).

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file to process. The file is read by TMIDIX.midi2single_track_ms_score.
    max_seq_len : int
        Maximum output tokens sequence length (truncated to this value). Default is 3072
    transpose_factor : int, optional
        Maximum semitone transpose range. The value is clamped to the inclusive range 0..6.
        If > 0, the function generates variants for transpositions in the integer range
        [-transpose_factor, transpose_factor - 1]. If 0, only the original (no transpose)
        variant is produced. Default is 6.
    verbose : bool, optional
        When True, prints concise progress messages and enables tqdm progress bars.
        Progress bars use `tqdm(disable=not verbose)` so they are suppressed when verbose is False.

    Returns
    -------
    list[list[int]]
        A list of token sequences. Each token sequence is a list of integers where:
        - The first element is 0 (start token).
        - Delta times (when nonzero) are appended as integers in 0..127.
        - Note events are encoded as two integers: duration_token and pitch_token,
          where duration_token = duration_clipped + 128 and pitch_token = pitch_clipped + 256.
        The function returns an empty list if processing fails or no notes are found.

    Notes and assumptions
    ---------------------
    - The function expects TMIDIX to provide the following functions used internally:
      midi2single_track_ms_score, advanced_score_processor, solo_piano_escore_notes,
      recalculate_score_timings, augment_enhanced_score_notes, remove_duplicate_pitches_from_escore_notes,
      fix_escore_notes_durations, delta_score_notes.
    - Delta events `d` are assumed to be indexable sequences where:
      d[1] is delta time, d[2] is duration, and d[4] is pitch (consistent with the original code).
    - Timing values are clipped to 0..127; durations are clipped to 1..127; pitches are clipped to 1..127
      after applying the transpose offset.
    - The function intentionally uses small integer ranges to match downstream token vocabularies
      that reserve offsets (e.g., +128, +256) for event encoding.

    Exceptions
    ----------
    - Any exception raised during processing is caught; the function prints a short error message
      (only when verbose) and returns the token sequences collected so far (often an empty list).

    Example
    -------
    >>> sequences = midi_to_tokens("example.mid", transpose_factor=2, verbose=True)
    >>> len(sequences)
    4  # variants for tv in [-2, -1, 0, 1] when transpose_factor == 2

    """
    
    all_toks_sequences = []

    try:
        if verbose:
            print('=' * 70)
            print(f"Loading MIDI file: {midi_file_path}")
            print('=' * 70)

        raw_score = TMIDIX.midi2single_track_ms_score(
            midi_file_path, do_not_check_MIDI_signature=True
        )

        if verbose:
            print("Running advanced score processor (enhanced notes, sustain applied)...")

        escore = TMIDIX.advanced_score_processor(
            raw_score, return_enhanced_score_notes=True
        )

        if not escore or not escore[0]:
            if verbose:
                print("No enhanced score notes found after advanced processing. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Extracting solo piano enhanced-score notes...")

        escore = TMIDIX.solo_piano_escore_notes(escore[0])

        if not escore:
            if verbose:
                print("Solo piano extraction returned no notes. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Recalculating timings, augmenting timings, removing duplicates, and fixing durations...")

        escore = TMIDIX.recalculate_score_timings(escore)
        escore = TMIDIX.augment_enhanced_score_notes(escore, timings_divider=32)
        escore = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore)

        # Clamp transpose_factor to allowed range
        transpose_factor = max(0, min(6, transpose_factor))
            
        if verbose:
            print(f"Using transpose_factor={transpose_factor} (clamped to 0..6).")

        if transpose_factor > 0:
            sidx = -transpose_factor
            eidx = transpose_factor
        else:
            sidx = 0
            eidx = 1

        if verbose:
            print(f"Generating token sequences for transpose variants in range({sidx}, {eidx})...")

        # Outer loop: transpose variants with progress bar
        for tv in tqdm.tqdm(range(sidx, eidx), disable=not verbose, desc="Transpose variants"):
            if verbose:
                print(f"Processing transpose variant tv={tv}...")

                chords = TMIDIX.escore_notes_to_chords(escore, shift_chords=True)

            all_toks_sequences.append(chords[:max_seq_len])

            if verbose:
                print(f"Variant tv={tv} produced sequence length {len(out_score[:max_seq_len])}.")

        if verbose:
            print('=' * 70)
            print(f"Finished processing. Produced {len(all_toks_sequences)} token sequence(s).")
            print('=' * 70)

        return all_toks_sequences

    except Exception as ex:
        print("Exception while converting MIDI to token sequences!")
        print(f"File: {midi_file_path}")
        print(f"Error: {ex}")
            
        return all_toks_sequences

###################################################################################

def idxs_sims_to_sorted_list(idxs: np.ndarray,
                             sims: np.ndarray,
                             sims_mult: int = 100,
                             remove_dupes=True,
                             ) -> List[Tuple]:
    
    """
    Helper function to convert indexes and similarities arrays into
    a sorted list with corresponding transpose values.
    
    Rwturns
    -------
    List of tuples: (corpus_index, transpose_value, cosine_similarity)
    """
    
    idxs = np.array(idxs)
    sims = np.array(sims)

    assert idxs.shape == sims.shape, f'Shape mismatch between indexes array and similarities array: {idxs.shape} != {sims.shape}'

    flat_idxs = [x for row in idxs.tolist() for x in row]
    flat_sims = [x * sims_mult for row in sims.tolist() for x in row]

    tv = idxs.shape[0]

    if tv == 1:
        sr = 0
        er = 1

    elif tv > 1 and tv % 2 == 0:
        sr = -(tv // 2)
        er = tv // 2

    else:
        sr = -6
        er = 6
    
    tkv = idxs.shape[1]
    
    flat_tvs = [v for v in range(sr, er) for _ in range(tkv)]

    sorted_list = sorted(zip(flat_idxs, flat_tvs, flat_sims), key=lambda x: -x[2])
    
    if remove_dupes:
        deduped_sorted_list = []
        seen = set()
        
        for idx, tv, sim in sorted_list:
            if idx not in seen:
                deduped_sorted_list.append([idx, tv, sim])
                seen.add(idx)
            
        return deduped_sorted_list
    
    return sorted_list   

###################################################################################

def print_sorted_idxs_sims_list(sorted_idxs_sims_list: list,
                                corpus_midi_names: Union[list, np.ndarray],
                                return_as_list: bool = False,
                                ) -> Union[List[Tuple], None]:
    
    """
    Helper function that prints search results list generated by idxs_sims_to_sorted_list function
    
    Returns
    -------
    List of tuples if return_as_list is True
    None if return_as_list is False
    """
    
    if type(corpus_midi_names) == np.ndarray:
        corpus_midi_names = corpus_midi_names.tolist()    

    if not return_as_list:
        print('=' * 70)
        print('Search results:')
        print('=' * 70)
    
    return_list = []

    for i, (idx, tv, sim) in enumerate(sorted_idxs_sims_list):

        if not return_as_list:
            print(f'#{str(i).zfill(3)} {corpus_midi_names[idx]} --- {tv} --- {round(sim, 8)}')
        
        else:
            return_list.append([i, corpus_midi_names[idx], tv, sim])    

    if not return_as_list:
        print('=' * 70)
        print('Total number of records:', len(sorted_idxs_sims_list))
        print('=' * 70)
    
    else:
        return return_list

###################################################################################

@lru_cache(maxsize=1)
def get_corpus_midis(corpus_midis_dirs_tuple: Tuple,
                     verbose: bool = True
                     ) -> Dict:
    
    """
    Returns corpus_midis_dic with LRU caching.
    corpus_midis_dirs_tuple must be a tuple for hashing.
    """

    if verbose:
        print("Scanning corpus MIDI directories...")

    # Create list
    corpus_midis_list = TMIDIX.create_files_list(
        list(corpus_midis_dirs_tuple),
        verbose=verbose
    )

    # Create dict: basename → full path

    if verbose:
        print('Converting files list to dict...')
        
    corpus_midis_dic = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in corpus_midis_list
    }

    if verbose:
        print('Done!')
    
    return corpus_midis_dic

###################################################################################

def copy_corpus_files(sorted_idxs_sims_list: list[list],
                      corpus_midis_dirs: list = ['./Corpus MIDIs Dir/'],
                      main_output_dir: str = './Corpus Matches Dir/',
                      sub_output_dir: str = 'Corpus MIDI Name',
                      copy_original_midi: bool = True,
                      verbose: bool = True
                     ) -> str:

    """
    Helper function that copies matched corpus MIDIs to a specified directory

    Returns
    -------
    Output directory where files were copied as a string
    """

    if verbose:
        print('=' * 70)
        print('Corpus MIDI files copier')
        print('=' * 70)

    if verbose:
        print('Creating corpus MIDIs files list dict...')

    corpus_midis_dic = get_corpus_midis(tuple(corpus_midis_dirs),
                                        verbose=verbose
                                       )
    
    if verbose:
        print('Done!')
        print('=' * 70)
        print('Copying files...')

    out_dir = ''
    original_copied = False

    for i, cfname, tv, sim in sorted_idxs_sims_list:
        
        try:
        
            sim = str(round(sim, 8))
            tv = str(tv)

            inp_fn = corpus_midis_dic[cfname]
    
            out_dir = os.path.join(main_output_dir, sub_output_dir)
            os.makedirs(out_dir, exist_ok=True)

            if copy_original_midi and not original_copied:
                
                src_fn = sub_output_dir + '.mid'
                out_src_fn = os.path.join(out_dir, src_fn)
                shutil.copy2(inp_fn, out_src_fn)
                original_copied = True
            
            out_fn = os.path.join(out_dir, sim + '_' + tv + '_' + cfname + '.mid')
    
            shutil.copy2(inp_fn, out_fn)

        except Exception as ex:
            if verbose:
                print(ex)
                print('Could not copy file #', i, ':', cfname)
                
            continue

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return out_dir

###################################################################################

def print_masked_predictions_ids(
    results,
    topk=1,
    mask_token='[M]',
    mask_idx=333,
    show_chords_intervals=True,
    show_chords_names=False
):
    """
    Prints aligned views for token-ID–based prediction results:
      • Original      (token IDs)
      • Masked        (token IDs or mask_token placeholder)
      • Predicted     (mask slots filled with colored top-1 IDs)
      • Reconstructed (mask slots filled with plain top-1 IDs)

    Then prints a per-position breakdown of top-k predictions by ID.

    Args:
        results: dict from predict_masked_tokens, with keys
                 'original_ids', 'masked_ids', 'predictions'
        topk: how many of the top predictions to list in the detail section
        mask_token: printed placeholder for a masked slot
        mask_idx: integer ID used to represent the mask in masked_ids
    """
    orig_ids   = results["original_ids"]
    masked_ids = results["masked_ids"]
    preds_map  = {p["position"]: p for p in results["predictions"]}
    n          = len(orig_ids)

    # 1) Prepare display tokens (as strings)
    original_tokens = [str(tid) for tid in orig_ids]
    masked_tokens   = [
        mask_token if tid == mask_idx else str(tid)
        for tid in masked_ids
    ]

    # 2) Build Predicted & Reconstructed views
    predicted_tokens   = []
    reconstructed_ids  = original_tokens.copy()

    for idx in range(n):
        if idx in preds_map:
            top1_id, top1_prob = preds_map[idx]["topk"][0]
            tok_str = str(top1_id)
            # green if correct, red if wrong
            color = "\033[92m" if top1_id == orig_ids[idx] else "\033[91m"
            predicted_tokens.append(f"{color}{tok_str}\033[0m")
            reconstructed_ids[idx] = tok_str
        else:
            predicted_tokens.append(masked_tokens[idx])

    # 3) Print the three aligned lines
    print(f"Original:      {' '.join(original_tokens)}")
    print(f"Masked:        {' '.join(masked_tokens)}")
    print(f"Predicted:     {' '.join(predicted_tokens)}")
    print(f"Reconstructed: {' '.join(reconstructed_ids)}\n")

    # 4) Detailed per-position breakdown
    print("Prediction Details:")
    for pos in sorted(preds_map):
        orig_id      = orig_ids[pos]

        if show_chords_intervals:
            if orig_id > 11:
                orig_id = TMIDIX.ALL_CHORDS_SORTED[orig_id-12]
                
            else:
                orig_id = [orig_id]

            if show_chords_names:
                orig_id = TMIDIX.get_chord_name(orig_id)
            
        pred_id, p = preds_map[pos]["topk"][0]

        if show_chords_intervals:
            if pred_id > 11:
                pred_id = TMIDIX.ALL_CHORDS_SORTED[pred_id-12]
    
            else:
                pred_id = [pred_id]

            if show_chords_names:
                pred_id = TMIDIX.get_chord_name(pred_id)
            
        status       = "\033[92m✓\033[0m" if pred_id == orig_id else "\033[91m✗\033[0m"
        print(f"Position {pos:2d}: {status}"
              f" Original {orig_id} → Predicted {pred_id} (P={p:.2%})")

        if topk > 1:
            for rank, (tid, prob) in enumerate(preds_map[pos]["topk"][:topk], start=1):
                print(f"    Top-{rank}: {tid} (P={prob:.2%})")
                
###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the midichords Python module
###################################################################################