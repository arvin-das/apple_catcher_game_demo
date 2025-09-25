import numpy as np
import pylsl
import mne
import pymatreader
from constants import *

def create_lsl_inlet(stream_name):
    streams = pylsl.resolve_stream('name',stream_name)
    inlet = pylsl.StreamInlet(streams[0])
    inlet.pull_sample()

    return inlet

def create_mne_info(inlet):
    n_channels = inlet.info().channel_count()

    if n_channels == 4:
        ch_names = CH_NAMES_4
    elif n_channels == 8:
        ch_names = CH_NAMES_8
    elif n_channels == 32:
        ch_names = CH_NAMES_32
    elif n_channels ==64:
        ch_names = CH_NAMES_64
    else:
        print(f"Number of channels not supported: {n_channels}")
        return None

    info = mne.create_info(ch_names=ch_names, sfreq=inlet.info().nominal_srate(), ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'))
    return info

def collect_data(inlet,inlet_info,offset):
    sfreq = inlet_info['sfreq']

    # Pull data from the given window
    sample,timestamps = inlet.pull_chunk(timeout = 0.0 , max_samples=int(sfreq*SAMPLE_WINDOW))
    sample = np.array(sample).transpose()
    timestamps = np.array(timestamps)+offset

    shape = sample.shape
    if shape!=(len(inlet_info['ch_names']),sfreq*SAMPLE_WINDOW):
        print(f"Sample was smaller than expected: {shape}")
    
    # Correct for out of order samples and jitter
    sample,timestamps = timestep_correction(sample,timestamps,out_of_order=True,dejitter=True)

    return sample,timestamps

def timestep_correction(sample,timestamps,out_of_order = True,dejitter = True):
    min = np.min(timestamps)
    timestamps = timestamps-min
    # Correct for samples out of order
    if out_of_order:
        correct_order = np.argsort(timestamps)
        sample = sample[:,correct_order]
        timestamps = timestamps[correct_order]
    # Correct for jitter. Current solution is very basic
    if dejitter:  
        max = np.max(timestamps)
        min = np.min(timestamps)
        timestamps = np.linspace(min,max,len(timestamps))
    return sample, timestamps

def clear_lsl_buffer(inlet):
    # Pull all remaining data in the buffer to clear it
    sample,timestamps = inlet.pull_chunk(timeout =0.0 , max_samples=10000000)
    return len(sample)

def load_giga_data(
    subject_number,
    set_average_reference=True,
    filter=None,
    baseline=None,
    reject=None,
    notch_filter=None,
    epochs_only=True,
    ):
    """
    Reads the raw data and creates epochs for a given subject number.

    Parameters
    ----------
    subject_number : int
        The subject number.
    set_average_reference : bool, optional
        Whether to set the average reference. The default is True.
    filter : tuple, optional
        The filter to apply. The default is None.
    baseline : tuple, optional
        The baseline to apply. The default is None.
    reject : dict, optional
        The rejection criteria. The default is None.
    notch_filter : float, optional
        The notch filter to apply. The default is None.
    epochs_only : bool, optional
        Whether to return only the epochs. The default is False.

    Returns
    -------
    raw_array_left : mne.io.RawArray
        The raw data for the left hand imagery.
    raw_array_right : mne.io.RawArray
        The raw data for the right hand imagery.
    epochs_left : mne.Epochs
        The epochs for the left hand imagery.
    epochs_right : mne.Epochs
        The epochs for the right hand imagery.
    """
    ch_types = ["eeg" for i in range(64)]
    rawdata = pymatreader.read_mat(
        "giga_mat_files/s" + str(subject_number).zfill(2)+".mat"
    )

    montage = mne.channels.make_standard_montage("biosemi64")
    info = mne.create_info(
        CH_NAMES_64,
        rawdata["eeg"]["srate"],
        ch_types=ch_types,
        verbose=False
    ).set_montage(montage)
    info["subject_info"] = dict(id=subject_number, his_id=rawdata["eeg"]["subject"])

    MI_left = rawdata["eeg"]["imagery_left"]
    MI_right = rawdata["eeg"]["imagery_right"]
    raw_array_left = mne.io.RawArray(MI_left[:64] * 1e-8, info, verbose=False)
    raw_array_right = mne.io.RawArray(MI_right[:64] * 1e-8, info, verbose=False)

    if set_average_reference:
        raw_array_left.set_eeg_reference(
            ref_channels="average", projection=True, verbose=False
        )
        raw_array_right.set_eeg_reference(
            ref_channels="average", projection=True, verbose=False
        )

    if filter is not None:
        raw_array_left.filter(filter[0], filter[1],verbose=False)
        raw_array_right.filter(filter[0], filter[1],verbose=False)

    if notch_filter is not None:
        raw_array_left.notch_filter(notch_filter, verbose=False)
        raw_array_right.notch_filter(notch_filter, verbose=False)

    imagery_event = rawdata["eeg"]["imagery_event"]
    event_id_left = {"MI_left": 0}
    event_id_right = {"MI_right": 1}
    event_indexes = []
    for i in range(len(imagery_event)):
        if imagery_event[i] == 1:
            event_indexes.append(i)
    events_left = np.array(
        [event_indexes, [0] * len(event_indexes), [0] * len(event_indexes)]
    ).T
    events_right = np.array(
        [event_indexes, [0] * len(event_indexes), [1] * len(event_indexes)]
    ).T

    tmin = -2
    tmax = 5

    epochs_left = mne.Epochs(
        raw_array_left,
        events_left,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_id=event_id_left,
        reject=reject,
        verbose=False
    )
    epochs_right = mne.Epochs(
        raw_array_right,
        events_right,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        event_id=event_id_right,
        reject=reject,
        verbose=False
    )

    bad_epochs_left = [
        x - 1
        for x in (
            list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][0])
            if isinstance(
                rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][0], list
            )
            else []
        )
        + (
            list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][0])
            if isinstance(
                rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][0], list
            )
            else []
        )
    ]

    bad_epochs_right = [
        x - 1
        for x in (
            list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][1])
            if isinstance(
                rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_mi"][1], list
            )
            else []
        )
        + (
            list(rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][1])
            if isinstance(
                rawdata["eeg"]["bad_trial_indices"]["bad_trial_idx_voltage"][1], list
            )
            else []
        )
    ]

    epochs_left.drop(bad_epochs_left, verbose=False)
    epochs_right.drop(bad_epochs_right, verbose=False)

    if epochs_only:
        return epochs_left, epochs_right
    else:
        return raw_array_left, raw_array_right, epochs_left, epochs_right