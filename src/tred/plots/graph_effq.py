#!/usr/bin/env python
from tred.graph import Drifter, Raster, ChunkSum, LacedConvo, Charge, Current, Sim
from tred.response import ndlarsim
from tred.blocking import Block, concat_blocks, iter_chunk_block
from tred import units
from .response import get_ndlarsim
from tred.util import debug, info, tenstr, warning, iter_tensor_chunks
from tred.loaders import StepLoader, steps_from_ndh5
from tred.io_nd import (
    nd_collate_fn, create_tpc_datasets_from_steps,
    LazyLabelBatchSampler, EagerLabelBatchSampler, SortedLabelBatchSampler, CustomNDLoader, simple_geo_parser
)
from tred.recombination import birks, box
from tred.io import write_npz
from tred import chunking
from tred.readout import nd_readout

import sys
import h5py
import numpy as np
import yaml
from collections import defaultdict
import torch
import time
import os

import torch
import time
import json
import mplhep as hep

# torch.float32 = torch.float64
# change float32 to float64 globally

# # set seed to 42 for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

module_yaml = None
tile_yaml = None
response_path = None
lifetime = None
input_path = None
output_path = None
drtoa = None
tspace = None
threshold = None
event_list = None
save_waveform = None
const_recomb = None

uncorr_noise = None
reset_noise = None
thres_noise = None
fluctuate = False
effq_out_nt = 1

# pitch = 4.434*units.mm / units.cm # values are in units of cm
# nimperpix=10 ## This is the default config we are using. Subdividing a pixel into 10x10 subpixels.
# nimperpix=6
# pspace = pitch/nimperpix
velocity = 1.59645 * units.mm/units.us / (units.cm/units.us) # values are in units of cm/us

adc_hold_delay = None
adc_down_time = None
csa_reset_time = None
one_tick = None

response = None

old_geo_config = True

def load_threshold(threshold):
    '''
    A map of io group from data to MC should be done.
    Hard-coded map is provided for 2x2 geometry.
    Assume thresholds are aligned from lower to high ends.
    '''
    if not isinstance(threshold, str):
        return [torch.tensor(threshold), ] * 1000 # FIXME: A large enough number
    thresholds = []
    # io_groups = [1, 2, 3, 4, 5, 6, 7, 8]
    # io_indices_tred = [1, 0, 3, 2, 5, 4, 7, 6]
    with h5py.File(threshold, 'r') as fthres:
        for ig in [2,1,4,3,6,5,8,7]:
            thresholds.append(torch.tensor(fthres[f'io_group{ig}/threshold']['Q'][:], dtype=torch.float32))
    return thresholds

def concatenate_waveforms(sparse_currents, Nt, event_t=0):
    '''
     Assume there is no overlap.
     Assume location is binned into 1x1 pixel groups.
     location: shape (Nbatch, vdim)
     data: shape (Nbatch, 1, 1, ,1, ..., Mt)
     Nt is the length of output along time axis (last axis). It must be divisible by Mt.
     '''
    data = sparse_currents.data
    location = sparse_currents.location
    Mt = data.shape[-1]
    if Nt % Mt:
        raise ValueError(f'Nt: {Nt} must be divisible by Mt: {Mt}')
    if any(l != 1 for l in data.shape[1:-1]):
        raise ValueError(f'Size in pixel domains must be 1, but {data.shape[1:-1]} is given.')

    vdim = location.shape[1]
    pixel_locs, rev_ind = torch.unique(location[:, :-1], dim=0, return_inverse=True, sorted=True)
    Npix = pixel_locs.size(0)
    Nbatch = location.shape[0]

    # Construct loc_out = [pixel_coords..., min_time_per_pixel]
    min_time = torch.full((Npix,), Nt+torch.max(location[:,-1]), device=location.device, dtype=location.dtype)
    min_time.scatter_reduce_(0, rev_ind, location[:, -1], reduce='amin', include_self=False)
    loc_out = torch.cat([pixel_locs, min_time.unsqueeze(1)], dim=1)  # shape (Npix, vdim)

    tref = min_time[rev_ind]

    # Initialize output waveform
    wf_out = torch.zeros((Npix, Nt), device=data.device, dtype=data.dtype)

    # Time assignment
    t_start = location[:, -1] - tref
    if torch.any(t_start<0):
        raise ValueError
    time_offsets = torch.arange(Mt, device=data.device).view(1, -1)
    time_indices = t_start.view(-1, 1) + time_offsets
    batch_indices = rev_ind.view(-1, 1).expand(-1, Mt).clone().detach()
    flat_batch = batch_indices.reshape(-1)
    flat_time = time_indices.reshape(-1)
    flat_values = data.view(-1, Mt).reshape(-1)
    wf_out.index_put_((flat_batch, flat_time), flat_values, accumulate=False)


    # Reshape waveform output to match original spatial dims
    wf_out = wf_out.view(Npix, *data.shape[1:-1], Nt)

    # filter negative ticks
    # Zero out any samples before event_t
    # For each pixel, if its start-time < event_t, zero samples index < event_t - start_time
    global_start = loc_out[:, -1]
    offsets = (event_t - global_start).clamp(min=0).to(torch.long)
    T = wf_out.size(-1)
    # mask_i,t = True if t < offsets[i]
    mask = (torch.arange(T, device=wf_out.device)[None, :] < offsets.to(wf_out.device)[:, None])
    # shape (Npix, T) -> insert singleton dims to cover the "..." in wf_out
    mask = mask.view(offsets.size(0), *([1] * (wf_out.ndim - 2)), T)
    # zero in-place where mask is True
    wf_out.masked_fill_(mask, 0)

    return Block(data=wf_out, location=loc_out)


def make_nd(device='cpu'):
    '''
    This mocks up some file of depo sets.
    '''

    borders = simple_geo_parser(module_yaml, tile_yaml, old_geo_config)
    d0 = StepLoader(h5py.File(input_path), transform=steps_from_ndh5)
    f0, f1, i0 = d0[:]
    # print('f0 ', f0, 'f1 ', f1, 'i0 ', i0)
    return (f0, f1, i0), i0, borders


def segment_to_tpc(features, labels, borders):
    tpcs = create_tpc_datasets_from_steps(features, labels, borders, sort_index=0)
    return tpcs


def transform_indices_to_coord_3d(location, pitch, tick, velocity,
                                  lower, anode, direction,
                                  paxes=(0,1), taxis=-1, offset=None):
    '''
    location: batched
    pitch: in cm
    tick: in us
    velocity: in cm/us
    lower: in cm
    anode: in cm
    direction: -1 or +1
    '''
    if offset is None:
        offset = torch.zeros((1,3,), dtype=torch.int32, device=location.device)
    locs = location.to(torch.float32) + offset
    locs[:,paxes] = locs[:,paxes].to(torch.float32)*pitch + lower.to(locs.device)
    locs[:,taxis] = anode - direction * velocity * tick * locs[:,taxis].to(torch.float32)
    return locs

def runit(device='cpu'):
    '''
    '''
    export_pickle = False

    # BATCH_SIZE = 2048 # 4096
    BATCH_SIZE = 4096 # select one segement at a time
    NBCHUNK = 100 # 100
    NBCHUNK_CONV = 100 # 50
    # eventually replace this hard-wire with configuration
    twindow_max = 12_000 # 12_000 * 50ns = 600us
    
    DL = 4.0 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    DT = 8.8 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    # DT = 10*8.8 * units.cm2/units.s / (units.cm2/units.us) # value are in cm2/us
    diffusion = torch.tensor([DL, DT, DT])
    grid_spacing = (pspace, pspace, tspace)
    npixpersuper = 12+1-9
    # ntickperslice = 6912+1-6400
    ntickperslice = 384 # 128*3
    chunk_shape = (npixpersuper * nimperpix, npixpersuper * nimperpix, ntickperslice)

    efield = 0.5 # kV/cm
    rho = 1.38 # g/cm^3
    A3t = 0.8 # birks
    k3t = 0.0486 # (g/MeV cm^2) (kV/cm); birks
    Wi = 23.6E-6 # MeV/pair

    lacing = torch.tensor([nimperpix, nimperpix, 1])

    batch_size = BATCH_SIZE

    t0 = time.time()
    # create intermediate nodes
    # dummy drifter for running time tests
    drifter = Drifter(diffusion, lifetime, velocity, drtoa=drtoa)
    raster = Raster(velocity, grid_spacing)
    chunksum = ChunkSum(chunk_shape)

    cshape_effq_out = torch.tensor([nimperpix, nimperpix, effq_out_nt])

    chunksum_effq_out = ChunkSum(cshape_effq_out) # 1 pixel, 1 pixel, 60*0.05us*1.6cm/us=4.8mm

    chunksum_readout = ChunkSum((1,1,120))
    # chunksum_readout = ChunkSum((1,1,12000))
    convo = LacedConvo(lacing, o_shape=(12, 12, 6912))
    # convo = LacedConvo(lacing, o_shape=(12, 12, 2048))
    chunksum_i = ChunkSum((4, 4, 128), method='chunksum_inplace_v2')

    chunksum_i = chunksum_i.to('cuda')
    chunksum_readout = chunksum_readout.to('cuda')
    chunksum_effq_out = chunksum_effq_out.to('cuda')

    t1 = time.time()

    # response = ndlarsim(response_path) # response is loaded in main function

    global response
    response = response.to(device=device)

    t2 = time.time()

    tpcs = segment_to_tpc(*make_nd('cpu'))

    t3 = time.time()

    runtime = defaultdict(list)

    waveforms = {}

    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000
    if export_pickle:
        torch.cuda.memory._record_memory_history(
        # max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    thresholds = load_threshold(threshold)

    # peak_memory_perTPC = {'batch_size': BATCH_SIZE,
    #                       'nbchunk': NBCHUNK,
    #                       'nbchunk_conv': NBCHUNK_CONV,}
    print('before TPC')

    drift_time = np.array([], dtype=np.float32)
    diffusion_Long_spread = np.array([], dtype=np.float32)
    diffusion_Transv_spread_x = np.array([], dtype=np.float32)
    diffusion_Transv_spread_y = np.array([], dtype=np.float32)
    for itpc, tpcdataset in enumerate(tpcs):
        # m0_start_tpc = torch.cuda.memory_allocated() / 1024**2
        info(f"Drift direction: {tpcdataset.drift} in tpcid {tpcdataset.tpc_id}.")
        info(f"TPC lower corner: {tpcdataset.lower_left_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC upper corner: {tpcdataset.upper_corner} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC anode: {tpcdataset.anode} in itpc {tpcdataset.tpc_id}.")
        info(f"TPC cathode: {tpcdataset.cathode} in itpc {tpcdataset.tpc_id}.")
        sampler = SortedLabelBatchSampler(tpcdataset.labels[:,0], batch_size)
        loader = CustomNDLoader(tpcdataset, sampler=sampler,
                                batch_size=None, collate_fn=nd_collate_fn)
        
        # m0_loader = torch.cuda.memory_allocated() / 1024**2

        drifter = Drifter(diffusion, lifetime, tpcdataset.drift*velocity, fluctuate=fluctuate,
                          target=tpcdataset.anode, drtoa=drtoa)
        drifter = drifter.to(device=device)
        
        # m0_drifter_init = torch.cuda.memory_allocated() / 1024**2

        raster = Raster(tpcdataset.drift*velocity, grid_spacing).to(device=device)
        
        # m0_raster_init = torch.cuda.memory_allocated() / 1024**2

        # raster = raster.to(device=device)
        chunksum = chunksum.to(device=device)
        # m0_chunksum_todevice = torch.cuda.memory_allocated() / 1024**2

        convo = convo.to(device=device)
        # m0_convo_todevice = torch.cuda.memory_allocated() / 1024**2

        tpc_lower_left = tpcdataset.lower_left_corner.to(device).unsqueeze(0)
        # m0_tpc_lower_left_todevice = torch.cuda.memory_allocated() / 1024**2

        ## Uncomment if you want to save output npz ------------------------------------------------
        waveforms[f'tpc_lower_left_tpc{tpcdataset.tpc_id}'] = tpc_lower_left.cpu().squeeze(0)
        waveforms[f'tpc_upper_tpc{tpcdataset.tpc_id}'] = tpcdataset.upper_corner.cpu()
        waveforms[f'drift_direction_tpc{tpcdataset.tpc_id}'] = tpcdataset.drift
        waveforms[f'tpc_anode_tpc{tpcdataset.tpc_id}'] = tpcdataset.anode
        waveforms[f'tpc_cathode_tpc{tpcdataset.tpc_id}'] = tpcdataset.cathode
        waveforms[f'pixel_pitch_tpc{tpcdataset.tpc_id}'] = pitch
        ## ---------------------------------------------------------------------------------


        inds_range = (tpcdataset.upper_corner - tpcdataset.lower_left_corner) // pitch
        inds_range = inds_range.to(torch.int32).to(device)
        # m0_inds_range = torch.cuda.memory_allocated() / 1024**2 # this was called m0_wf_init in the older version of the code : tred_2

        # peak_memory_perTPC[f'tpc{itpc}'] = {
        #     'start_tpc_MB': m0_start_tpc,
        #     'loader_init_MB': m0_loader,
        #     'drifter_init_MB': m0_drifter_init,
        #     'raster_init_MB': m0_raster_init,
        #     'chunksum_todevice_MB': m0_chunksum_todevice,
        #     'convo_todevice_MB': m0_convo_todevice,
        #     'tpc_lower_left_todevice_MB': m0_tpc_lower_left_todevice,
        #     'inds_range_todevice_MB': m0_inds_range
        # }

        peak_memory_perbatch = {}

        for ibatch, (features, labels) in enumerate(loader):

            stime = time.time()
            try:
                if isinstance(event_list, list) and len(event_list)>0 and int(labels[0,0].numpy()) not in event_list:
                    continue
                
                global_tref = [features[0][0,-2].numpy(), torch.min(features[0][:,-1]).numpy()] # assume it is in us
                # print(f'global_tref : {global_tref[0].shape}')
                # print(features[0][0].shape, features[0][0, -2], features[0][0][-2])
                # print(features[0][:, -1])
                # sys.exit()
                ## Uncomment if you want to save output npz ------------------------------------------------
                waveforms[f'global_tref_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = np.array(global_tref)
                waveforms[f'event_id_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = labels[0,0].numpy()
                # assume there is only one particle in the event
                waveforms[f'event_start_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = features[0][0,2:5].numpy()
                waveforms[f'event_end_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = features[0][-1,5:8].numpy()
                ## ---------------------------------------------------------------------------------
                
                # if device == 'cuda':
                #     torch.cuda.synchronize()
                # t00 = time.time()
                features = [f.to(device=device) for f in features]

                # if device == 'cuda':
                #     torch.cuda.synchronize()
                # t01 = time.time()

                charge = birks(dE=features[0][:,0], dEdx=features[0][:,1],
                          efield=efield, rho=rho, A3t=A3t, k3t=k3t, Wi=Wi)
                if const_recomb:
                    charge = features[0][:,0] / Wi * const_recomb # MeV / MeV/pair
                # mem_recomb = torch.cuda.memory_allocated() / 1024**2

                if device == 'cuda':
                    torch.cuda.synchronize()
                t02 = time.time()

                local_time = features[0][:,-1]
                tail = features[0][:,2:5]
                head = features[0][:,5:8]
                tail[:,[1,2]] -= tpc_lower_left
                head[:,[1,2]] -= tpc_lower_left
                
                # print(f'features[0] shape: {features[0].shape}, tail : {tail.shape}, head : {head.shape}, local_time : {local_time.shape}, charge : {charge.shape}')
                # print(f'features[0][0] shape : {features[0][0].shape}')
                # print(f'global time : {features[0][0,-2]}, local time : {features[0][0,-1]}')
                # sys.exit()
                ### apply a cut on the time offset
                # mask = local_time >= 5 #>= 5# <= 3
                # local_time = local_time[mask]
                # tail = tail[mask]
                # head = head[mask]
                # charge = charge[mask]
                # dsigma, dtime, dcharge, dtail, dhead
                drifted = drifter(local_time, charge, tail, head)
                tdrift = drifted[1] + torch.abs(drtoa / (tpcdataset.drift*velocity)) - local_time
                # print(f'tdrift : {tdrift} us, Negative tdrift : {(tdrift<0).sum().item()} out of {tdrift.shape[0]} points.')  # DEBUG
                if (len(drift_time) == 0):
                    drift_time = tdrift.cpu().numpy()
                else:
                    drift_time = np.concatenate((drift_time, tdrift.cpu().numpy()), axis=0)
                # continue
                # print(f'Drift time : {tdrift} us, Negative drift time : {(tdrift<0).sum().item()} out of {tdrift.shape[0]} points.')  # DEBUG
                # continue
                # m1_drifter = torch.cuda.memory_allocated() / 1024**2
                # print(f'local time : {local_time}')
                # print(f'drift time : {drifted[1]}')
                # sys.exit()
                # dsigma, dtime, dcharge, dtail, dhead = drifter(local_time, charge, tail, head)
                ## Uncomment if you need runtime -------------------------------------------------
                if device == 'cuda':
                    torch.cuda.synchronize()
                t03 = time.time()
                ## ---------------------------------------------------------------------------------

                nbchunk = NBCHUNK

                current_blocks = []
                effq_blocks = []
                Nqblock = 0
                
                mem_usage_chunking = {}

                for ichunk, idrifted in enumerate(
                        iter_tensor_chunks(drifted, chunk_size=nbchunk)):
                    # if len(drift_time) == 0:
                    #     drift_time = idrifted[1].cpu().numpy()
                    #     continue
                    # else:
                    #     drift_time = np.concatenate((drift_time, idrifted[1].cpu().numpy()), axis=0)
                    #     continue

                    # qblock = raster(*idrifted)
                    # mask_early = idrifted[1] < 30 ## cut on drift time > 100 us 
                    # idrifted = tuple([x[mask_early] for x in idrifted])
                    # mask_neg_tdrift = idrifted[1] < 0
                    # idrifted = tuple([x[mask_neg_tdrift] for x in idrifted])

                    if len(diffusion_Long_spread) == 0:
                        # drift_time = idrifted[1].cpu().numpy()
                        diffusion_Long_spread = idrifted[0][:, 0].cpu().numpy()
                        diffusion_Transv_spread_x = idrifted[0][:, 1].cpu().numpy()
                        diffusion_Transv_spread_y = idrifted[0][:, 2].cpu().numpy()
                        continue
                    else:
                        # drift_time = np.concatenate((drift_time, idrifted[1].cpu().numpy()), axis=0)
                        diffusion_Long_spread = np.concatenate((diffusion_Long_spread, idrifted[0][:, 0].cpu().numpy()), axis=0)
                        diffusion_Transv_spread_x = np.concatenate((diffusion_Transv_spread_x, idrifted[0][:, 1].cpu().numpy()), axis=0)
                        diffusion_Transv_spread_y = np.concatenate((diffusion_Transv_spread_y, idrifted[0][:, 2].cpu().numpy()), axis=0)
                        continue

                    qblock = raster(*idrifted, npoints=NPOINTS) # include the number of nodes for the quadrature rule
                    # mem_end_raster = torch.cuda.memory_allocated() / 1024**2

                    start = ichunk * nbchunk
                    end = start + idrifted[0].size(0)
                    p0 = tail[start:end]
                    p1 = head[start:end]
                    length2 = torch.sum((p0-p1)**2, dim=1)
                    invalid2 = length2 < 1E-9

                    qblock.data[invalid2] = 0

                    signal = chunksum(qblock)
                    # mem_chunksum_qblock = torch.cuda.memory_allocated() / 1024**2
                    ## Uncomment if you want to save output npz ------------------------------------------------
                    effqb = chunksum_effq_out(qblock)
                    effqb.location[:, 0:2] //= nimperpix
                    effqb.location[:, -1] += int(abs(drtoa/velocity)//tspace)
                    effq_blocks.append(effqb)
                    qblock = None
                    effqb = None
                    ## ---------------------------------------------------------------------------------
                    Nqblock += signal.nbatches

                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t05 = time.time()
                    # chunk_conv_mem = {}
                    currents = []
                    ichunk_conv = 0
                    for iqblock in iter_chunk_block(signal, chunk_size=NBCHUNK_CONV):
                        if iqblock.nbatches == 0:
                            continue
                        iblock = convo(iqblock, response)
                        # m2 = torch.cuda.memory_allocated() / 1024**2

                        current = chunksum_i(iblock)
                        # m3 = torch.cuda.memory_allocated() / 1024**2
                        currents.append(current)
                        # chunk_conv_mem[f'ichunk_conv{ichunk_conv}'] = {
                        #     'conv_MB': m2,
                        #     'chunksum_i_MB': m3
                        # }
                        ichunk_conv += 1

                    # mem_end_conv = torch.cuda.memory_allocated() / 1024**2

                    # no need to chunk again; just sum
                    currents = concat_blocks(currents)
                    if currents is not None:
                        currents = chunking.accumulate(currents)
                        current_blocks.append(currents)
                    # mem_end_sumcurrent = torch.cuda.memory_allocated() / 1024**2

                    # mem_usage_chunking[f'ichunk_{ichunk}'] = {
                    #     'raster_MB': mem_end_raster,
                    #     'chunksum_qblock_MB': mem_chunksum_qblock,
                    #     'conv_MB': {
                    #         'total_MB': mem_end_conv,
                    #         'details': chunk_conv_mem
                    #     },
                    #     'sumcurrent_MB': mem_end_sumcurrent
                    # }
                    # if device == 'cuda':
                    #     torch.cuda.synchronize()
                    # t05 = time.time()
                continue ## just skip

                ## Uncomment if you want to save output npz ------------------------------------------------
                effq_blocks = concat_blocks(effq_blocks, device='cpu')
                print(f'SHAPE of effq blocks after concat: {effq_blocks.data.shape}')
                ## ---------------------------------------------------------------------------------

                # no need to chunk again; just sum
                currents = concat_blocks(current_blocks)
                if currents is not None:
                    currents = chunking.accumulate(currents)
                # mem_sumcurrent = torch.cuda.memory_allocated() / 1024**2

                ## Uncomment if you need runtime -------------------------------------------------
                t04 = t03
                t05 = t04
                if device == 'cuda':
                    torch.cuda.synchronize()
                t06 = time.time()

                if device == 'cuda':
                    torch.cuda.synchronize()
                t07 = time.time()
                
                if currents is None:
                    info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                         f'N segments {len(features[0])}, '
                         f'N qblock {Nqblock}, '
                         f'elapsed {t07 - stime} sec on {device}. Skipped empty batch.')
                    continue
                ## ---------------------------------------------------------------------------------
                # if currents is None:
                #     continue

                currents = chunksum_readout(currents)
                # mem_readout_chunksum = torch.cuda.memory_allocated() / 1024**2
                currents = concatenate_waveforms(currents, twindow_max, event_t=global_tref[1]//tspace)
                # mem_readout_concat = torch.cuda.memory_allocated() / 1024**2

                currents.data = currents.data * tspace / 1E3 # to ke- 
                current_mask = (currents.location[:,[0,1]] <= inds_range) & (currents.location[:,[0,1]] >= 0)
                current_mask = current_mask.all(dim=1)
                currents = Block(data=currents.data[current_mask], location=currents.location[current_mask])
                # mem_readout_current = torch.cuda.memory_allocated() / 1024**2
                # mem_each_operation = {
                #     'recomb_MB': mem_recomb,
                #     'drifter_MB': m1_drifter,
                #     'chunking_conv': mem_usage_chunking,
                #     'sum_current_MB': mem_sumcurrent,
                #     'chunksum_readout_MB': mem_readout_chunksum,
                #     'concat_readout_MB': mem_readout_concat,
                #     'formingBlock_readout_current_MB': mem_readout_current
                # }

                # peak_memory_perbatch[f'batch_label{ibatch}'] = {
                #     'event_id': int(labels[0,0].numpy()),
                #     'N_segments': len(features[0]),
                #     'N_qblock': Nqblock,
                #     'peak_memory_MB': torch.cuda.max_memory_allocated() / 1024**2,
                #     'each_operation_MB': mem_each_operation
                # }
                # torch.cuda.reset_peak_memory_stats()
                ## Uncomment if you want to save output npz ------------------------------------------------
                if torch.isnan(currents.data).any():
                    # raise ValueError
                    info(ValueError)

                # if isinstance(threshold, str):
                #     raise NotImplementedError("To add support for loading a threshold file.")
                thres = thresholds[tpcdataset.tpc_id].to(device)
                if thres.ndim > 0:
                    thres[thres<2] = 1E16 # FIXME: Temporarily disable low threshold channels
                hits = nd_readout(currents, thres, adc_hold_delay, adc_down_time, csa_reset_time, one_tick=one_tick,
                                  offset_to_align=0, # FIXME: how to calculate properly?
                                  pixel_axes=(1,2), uncorr_noise=uncorr_noise, thres_noise=thres_noise, reset_noise=reset_noise)
                
                # runtime['to_device'].append(t01-t00)
                # runtime['recomb'].append(t02-t01)
                # runtime['drift'].append(t03-t02)
                # runtime['raster'].append(t04-t03)
                # runtime['chunksum_charge'].append(t05-t04)
                # runtime['convo'].append(t06-t05)
                # runtime['chunksum_current'].append(t07-t06)

                # info(f'{runtime["to_device"][-1]} data to {device}')
                # info(f'{runtime["recomb"][-1]} recomb')
                # info(f'{runtime["drift"][-1]} drift')
                # info(f'{runtime["raster"][-1]} raster')
                # info(f'{runtime["chunksum_charge"][-1]} chunksum_charge')
                # info(f'{runtime["convo"][-1]} convo')
                # info(f'{runtime["chunksum_current"][-1]} chunksum_current')

                if device == 'cuda':
                    cuda_mem = torch.cuda.max_memory_allocated() / 1024**2
                    info(f'Peak cuda usage: {cuda_mem} MB')

                info(f'itpc{itpc}, tpc label {tpcdataset.tpc_id}, batch label {ibatch}, '
                      f'N segments {len(features[0])}, '
                      f'N qblock {Nqblock}, '
                      f'elapsed {t07 - stime} sec on {device}.')

                if save_waveform and currents is not None:
                    waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = currents.data.cpu().numpy()
                    waveforms[f'current_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = currents.location.cpu().numpy()

                # FIXME: global time offset
                qbl = effq_blocks.location.to('cpu')
                qoff = cshape_effq_out / 2
                qoff[[0,1]] = qoff[[0,1]] / nimperpix
                qoff[2] -= global_tref[1]//tspace
                qblf32 = transform_indices_to_coord_3d(qbl, pitch, tspace, velocity,
                                                       tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                       paxes=(0,1), taxis=-1, offset=qoff)
                # print(f'qblf32 sample: {qblf32[0]}')
                qblf32 = qblf32[:, [2,0,1]]
                # print(f'qblf32 reordered (coord): {qblf32[0]}')
                qbd_fg = effq_blocks.data / 1E3 # to ke-
                # print(f'shape of qbg fg : {qbd_fg.shape}')
                qbd = qbd_fg.sum(dim=(1,2,3))
                # print(f'qbd [0] : {qbd[0]}')
                # print(f'shape of qbd coarse : {qbd.shape}')
                qbd = torch.cat([qblf32, qbd[:,None]], dim=1)
                # print(f'shape of qbg coarse with loc : {qbd.shape}')
                # print(f'qbd coarse with loc sample: {qbd[0]}')
                hitl = hits[0].cpu()
                # FIXME: :,:3 is hard-coded
                hoff = torch.tensor([1/2, 1/2, adc_hold_delay-global_tref[1]//tspace]).to(torch.float32)
                hitlf32 = transform_indices_to_coord_3d(hitl[:,:3], pitch, tspace, velocity,
                                                        tpc_lower_left.to(torch.float32), tpcdataset.anode, tpcdataset.drift,
                                                        paxes=(0,1), taxis=-1, offset=hoff)
                
                hitlf32 = hitlf32[:, [2,0,1]]
                hitd = torch.cat([hitlf32, hits[1][:,None].cpu()], dim=1)

                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = hitd.numpy() # check the type
                waveforms[f'hits_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = hitl.numpy()
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd
                print('----------------------------------------------------------------')
                print(f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch} shape : {qbd.shape}')
                print(f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch} sample : {qbd[0]}')
                print(f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location shape : {qbl.shape}')
                print(f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location sample : {qbl[0]}')
                print('----------------------------------------------------------------')
                waveforms[f'effq_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl
                waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}'] = qbd_fg
                waveforms[f'effq_fine_grain_tpc{tpcdataset.tpc_id}_batch{ibatch}_location'] = qbl

                torch.cuda.reset_peak_memory_stats()
                ## ---------------------------------------------------------------------------------------
            except IndexError as e:
                # raise e
                info(e)
            except Exception as e:
                info(f'Failed to process the batch {ibatch}')
                info(e)
        # peak_memory_perTPC[f'tpc{itpc}']['peak_memory_perbatch'] = peak_memory_perbatch
    import matplotlib.pyplot as plt
    hep.style.use("CMS") 
    ## Distribution of the drift time
    plt.figure()
    plt.hist(drift_time, bins=100, histtype='step', linewidth=2)
    # plt.yscale('log')
    plt.xlabel('Drift time (us)')
    plt.ylabel('Counts')
    plt.title('Drift time distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/CORRECT_drift_time_distribution.png')
    plt.close()
    ## Distribution of the diffusion spread
    plt.figure()
    plt.hist(diffusion_Long_spread, histtype='step', bins=100, label='Longitudinal spread', linewidth=2)
    plt.hist(diffusion_Transv_spread_x, histtype='step', bins=100, label='Transverse spread', alpha=0.7, linewidth=2)
    plt.xlabel('Diffusion spread (cm)')
    plt.ylabel('Counts')
    plt.title('Diffusion spread distribution')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/CORRECT_diffusion_spread_distribution.png')
    plt.close()
    ## 2d correlation plot of the drift time vs diffusion spread
    ## longitudinal vs transverse spreads
    plt.figure()
    plt.hist2d(diffusion_Long_spread, diffusion_Transv_spread_x, bins=100, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Counts')
    plt.xlabel('Longitudinal spread (cm)')
    plt.ylabel('Transverse spread (cm)')
    plt.title('2D Correlation: \nLongitudinal vs Transverse Diffusion Spread')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/CORRECT_2d_correlation_longitudinal_vs_transverse_diffusion_spread.png')
    plt.close()
    ## t_drift vs longitudinal spread
    plt.figure()
    plt.hist2d(drift_time, diffusion_Long_spread, bins=100, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Counts')
    plt.xlabel('Drift time (us)')
    plt.ylabel('Longitudinal spread (cm)')
    plt.title('2D Correlation: \nDrift Time vs Longitudinal Diffusion Spread')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/CORRECT_2d_correlation_drift_time_vs_longitudinal_diffusion_spread.png')
    plt.close()
    ## t_drift vs transverse spread
    plt.figure()
    plt.hist2d(drift_time, diffusion_Transv_spread_x, bins=100, cmap='viridis', norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(label='Counts')
    plt.xlabel('Drift time (us)')
    plt.ylabel('Transverse spread (cm)')
    plt.title('2D Correlation: \nDrift Time vs Transverse Diffusion Spread')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/ACC_EFFQ/CORRECT_2d_correlation_drift_time_vs_transverse_diffusion_spread.png')
    plt.close()
    sys.exit()
    # Stop recording memory snapshot history.
    ## Uncomment if you want to save the output -------------
    waveforms["tile_yaml"] = tile_yaml
    waveforms["module_yaml"] = module_yaml
    waveforms["response_path"] = response_path
    waveforms["lifetime"] = lifetime
    waveforms["drtoa"] = drtoa
    waveforms["threshold"] = threshold
    waveforms["event_list"] = event_list
    waveforms["save_waveform"] = save_waveform
    waveforms["uncorr_noise"] = uncorr_noise
    waveforms["thres_noise"] = thres_noise
    waveforms["reset_noise"] = reset_noise
    waveforms["fluctuate"] = fluctuate
    waveforms["effq_out_nt"] = effq_out_nt
    waveforms["input_path"] = input_path
    waveforms["adc_hold_delay"] = adc_hold_delay
    waveforms["adc_down_time"] = adc_down_time
    waveforms["csa_reset_time "] = csa_reset_time
    waveforms["one_tick"] = one_tick
    waveforms[f'time_spacing'] = tspace

    write_npz(output_path, **waveforms)
    ## -----------------------------------------------------
    
    info(f'{t1-t0} construct')
    info(f'{t2-t1} get response')
    info(f'{t3-t2} load nd from disk')
    info(f'{sum(runtime["to_device"])} data to {device}')
    info(f'{sum(runtime["recomb"])} recomb')
    info(f'{sum(runtime["drift"])} drift')
    info(f'{sum(runtime["raster"])} raster')
    info(f'{sum(runtime["chunksum_charge"])} chunksum_charge')
    info(f'{sum(runtime["convo"])} convo')
    info(f'{sum(runtime["chunksum_current"])} chunksum_current')

    info(f'Total elapsed time {time.time() - t0} seconds')

    try:
        if export_pickle:
            torch.cuda.memory._dump_snapshot(f"graph_effq.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

    info(f"Peak memory usage {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    ##
    ## save peak memory usage per TPC and per batch
    # with open(f'/home/rrazakami/work/ND-LAr/starting_over/OUTPUT_EVAL/MEMORY_EVAL/peak_memory_usage_2.json', 'w') as fpm:
    #     json.dump(peak_memory_perTPC, fpm)

def plots(out):
    with torch.no_grad():
        # torch.set_default_device('cuda')
        # runit('cpu')
        # info('FINISHED CPU')
        runit('cuda')
        info('FINISHED CUDA')

def fullsim(config, finpath, foutpath):

    global tile_yaml
    global module_yaml
    global response_path
    global lifetime
    global drtoa
    global tspace
    global threshold
    global event_list
    global save_waveform
    global uncorr_noise
    global thres_noise
    global reset_noise
    global fluctuate
    global effq_out_nt
    global adc_hold_delay
    global adc_down_time
    global csa_reset_time
    global one_tick

    global const_recomb

    global old_geo_config

    global input_path
    global output_path

    global response

    ##------ RADO
    global nimperpix
    global nd_response_shape
    global pitch
    global pspace
    global NPOINTS
    ## ----------------
    with open(config, "r") as fconfig:
        config = yaml.safe_load(fconfig)

    tile_yaml = config.get('tile_yaml',  "tests/playground/multi_tile_layout-2.4.16.yaml")
    module_yaml = config.get('module_yaml',  "tests/playground/2x2_mod2mod_variation.yaml")
    response_path = config.get("response_path",  "response_v2a_distance_10p431cm_binsize_0p04434cm_tick0p05us.npy")
    drtoa = config.get("drtoa", 10.431) * units.cm / units.cm # values are in units of cm to cm
    tspace = config.get("tspace", 0.05) * units.us/ units.us # values are in units of us
    lifetime = config.get("lifetime", 2.0) * units.ms / units.us # values are from ms units of us
    threshold = config.get("threshold", 5_000) # electrons # it can also be a path to threshold
    event_list = config.get("event_list", None) # None means select all
    save_waveform = config.get("save_waveform", False)
    uncorr_noise = config.get("uncorr_noise", None)
    thres_noise = config.get("thres_noise", None)
    reset_noise = config.get("reset_noise", None)
    fluctuate = config.get("fluctuate", False)
    const_recomb = config.get("const_recomb", False)
    effq_out_nt = config.get("effq_out_nt", 1)
    old_geo_config = config.get("old_geo_config", True)
    ## --- RADO ----
    nimperpix = config.get('nimperpix', 10)
    nd_response_shape = config.get('nd_response_shape', list([45, 45,]))

    pitch = 4.434*units.mm / units.cm # values are in units of cm
    # nimperpix=6
    pspace = pitch/nimperpix
    NPOINTS = config.get('npoints', (2,2,2))
    NPOINTS = tuple(NPOINTS)
    # ----------
    # get all events
    # event_list = None
    # loading response
    if os.path.splitext(response_path)[1] == '.npz':
        fres = np.load(response_path)
        response = ndlarsim(fres['response'], nd_nimp=nimperpix, nd_response_shape=nd_response_shape)
        tspace = fres['time_tick']  * units.us / units.us # us
        drtoa = fres['drift_length'] * units.cm / units.cm # cm
        bin_size = fres["bin_size"] * units.cm / units.cm # cm
        warning(f'drtoa, tspace, will be overridden to {drtoa} cm, {tspace} us.')
        if abs(bin_size - pspace) > 1E-4:
            warning(f'Please manually check pspace. pspace in response file is {fres["bin_size"]} cm. pspace in config.')
    else:
        response = ndlarsim(response_path, nd_nimp=nimperpix, nd_response_shape=nd_response_shape)

    adc_hold_delay = config.get("adc_hold_delay", 1.5) * units.us / units.us / (tspace * units.us / units.us)
    adc_hold_delay = int(round(adc_hold_delay))
    adc_down_time = config.get("adc_down_time", 1.2) * units.us / units.us / (tspace * units.us / units.us)
    adc_down_time = int(round(adc_down_time))
    csa_reset_time = config.get("csa_reset_time", 0.1) * units.us / units.us / (tspace * units.us / units.us)
    csa_reset_time = int(round(csa_reset_time))
    one_tick = config.get("one_tick", 0.1) * units.us / units.us / (tspace * units.us / units.us)
    one_tick = int(round(one_tick))

    if finpath is None:
        input_path = "/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/filtered_MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5"
    else:
        input_path = finpath

    if foutpath is None:
        output_path = "waveforms.npz"
    else:
        output_path = foutpath

    with torch.no_grad():
        runit('cuda')
