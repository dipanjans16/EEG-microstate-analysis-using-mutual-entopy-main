# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:24:38 2022

@author: BioImaging

"""


import argparse, os, sys, time
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt, welch
from scipy.stats import chi2, chi2_contingency
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from statsmodels.stats.multitest import multipletests
# in case you have a compiled Cython implementation:
# from mutinf.mutinf_cy import mutinf_cy_c_wrapper, mutinf_cy


def r_xyz(a):
    """Read EEG electrode locations in xyz format

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    ch_names = []
    locs = []
    ch_names=['Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P7','P8','P3','P4','O1','O2','Fz','Cz','Pz']
    locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-4.7,  6.2,  8. ],[ 4.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-4.7, -6.2,  8. ],[ 4.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
    return ch_names, np.array(locs)


def read_edf(filename):
    """Basic EDF file format reader

    EDF specifications: http://www.edfplus.info/specs/edf.html

    Args:
        filename: full path to the '.edf' file
    Returns:
        chs: list of channel names
        fs: sampling frequency in [Hz]
        data: EEG data as numpy.array (samples x channels)
    """

    def readn(n):
        """read n bytes."""
        return np.fromfile(fp, sep='', dtype=np.int8, count=n)

    def bytestr(bytes, i):
        """convert byte array to string."""
        return np.array([bytes[k] for k in range(i*8, (i+1)*8)]).tostring()

    fp = open(filename, 'r')
    x = np.fromfile(fp, sep='', dtype=np.uint8, count=256).tostring()
    header = {}
    header['version'] = x[0:8]
    header['patientID'] = x[8:88]
    header['recordingID'] = x[88:168]
    header['startdate'] = x[168:176]
    header['starttime'] = x[176:184]
    header['length'] = int(x[184:192]) # header length (bytes)
    header['reserved'] = x[192:236]
    header['records'] = int(x[236:244]) # number of records
    header['duration'] = float(x[244:252]) # duration of each record [sec]
    header['channels'] = int(x[252:256]) # ns - number of signals
    n_ch = header['channels']  # number of EEG channels
    header['channelname'] = (readn(16*n_ch)).tostring()
    header['transducer'] = (readn(80*n_ch)).tostring().split()
    header['physdime'] = (readn(8*n_ch)).tostring().split()
    header['physmin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['physmin'].append(float(bytestr(b, i)))
    header['physmax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['physmax'].append(float(bytestr(b, i)))
    header['digimin'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['digimin'].append(int(bytestr(b, i)))
    header['digimax'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['digimax'].append(int(bytestr(b, i)))
    header['prefilt'] = (readn(80*n_ch)).tostring().split()
    header['samples_per_record'] = []
    b = readn(8*n_ch)
    for i in range(n_ch):
        header['samples_per_record'].append(float(bytestr(b, i)))
    nr = header['records']
    n_per_rec = int(header['samples_per_record'][0])
    n_total = int(nr*n_per_rec*n_ch)
    fp.seek(header['length'],os.SEEK_SET)  # header end = data start
    data = np.fromfile(fp, sep='', dtype=np.int16, count=n_total)  # count=-1
    fp.close()

    # re-order
    #print("EDF reader:")
    #print("[+] n_per_rec: {:d}".format(n_per_rec))
    #print("[+] n_ch: {:d}".format(n_ch))
    #print("[+] nr: {:d}".format(nr))
    #print("[+] n_total: {:d}".format(n_total))
    #print(data.shape)
    data = np.reshape(data,(n_per_rec,n_ch,nr),order='F')
    data = np.transpose(data,(0,2,1))
    data = np.reshape(data,(n_per_rec*nr,n_ch),order='F')

    # convert to physical dimensions
    for k in range(data.shape[1]):
        d_min = float(header['digimin'][k])
        d_max = float(header['digimax'][k])
        p_min = float(header['physmin'][k])
        p_max = float(header['physmax'][k])
        if ((d_max-d_min) > 0):
            data[:,k] = p_min+(data[:,k]-d_min)/(d_max-d_min)*(p_max-p_min)

    #print(header)
    return header['channelname'].split(),\
           header['samples_per_record'][0]/header['duration'],\
           data


def bp_filter(data, f_lo, f_hi, fs):
    """Digital band pass filter (6-th order Butterworth)

    Args:
        data: numpy.array, time along axis 0
        (f_lo, f_hi): frequency band to extract [Hz]
        fs: sampling frequency [Hz]
    Returns:
        data_filt: band-pass filtered data, same shape as data
    """
    data_filt = np.zeros_like(data)
    f_ny = fs/2.  # Nyquist frequency
    b_lo = f_lo/f_ny  # normalized frequency [0..1]
    b_hi = f_hi/f_ny  # normalized frequency [0..1]
    # band-pass filter parameters
    p_lp = {"N":6, "Wn":b_hi, "btype":"lowpass", "analog":False, "output":"ba"}
    p_hp = {"N":6, "Wn":b_lo, "btype":"highpass", "analog":False, "output":"ba"}
    bp_b1, bp_a1 = butter(**p_lp)
    bp_b2, bp_a2 = butter(**p_hp)
    data_filt = filtfilt(bp_b1, bp_a1, data, axis=0)
    data_filt = filtfilt(bp_b2, bp_a2, data_filt, axis=0)
    return data_filt


def plot_data(data, fs):
    """Plot the data

    Args:
        data: numpy.array
        fs: sampling frequency [Hz]
    """
    t = np.arange(len(data))/fs # time axis in seconds
    fig = plt.figure(1, figsize=(20,4))
    plt.plot(t, data, '-k', linewidth=1)
    plt.xlabel("time [s]", fontsize=24)
    plt.ylabel("potential [$\mu$V]", fontsize=24)
    plt.tight_layout()
    plt.show()


def plot_psd(data, fs, n_seg=1024):
    """Plot the power spectral density (Welch's method)

    Args:
        data: numpy.array
        fs: sampling frequency [Hz]
        n_seg: samples per segment, default=1024
    """
    freqs, psd = welch(data, fs, nperseg=n_seg)
    fig = plt.figure(1, figsize=(16,8))
    plt.semilogy(freqs, psd, 'k', linewidth=3)
    #plt.loglog(freqs, psd, 'k', linewidth=3)
    #plt.xlim(freqs.min(), freqs.max())
    #plt.ylim(psd.min(), psd.max())
    plt.title("Power spectral density (Welch's method  n={:d})".format(n_seg))
    plt.tight_layout()
    plt.show()


def topo(data, n_grid=64):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    channels=['Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P7','P8','P3','P4','O1','O2','Fz','Cz','Pz']
    locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-4.7,  6.2,  8. ],[ 4.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-4.7, -6.2,  8. ],[ 4.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
    n_channels = len(channels)
    #locs /= np.sqrt(np.sum(locs**2,axis=1))[:,np.newaxis]
    locs /= np.linalg.norm(locs, 2, axis=1, keepdims=True)
    c = findstr('Cz', channels)[0]
    # print 'center electrode for interpolation: ' + channels[c]
    #w = np.sqrt(np.sum((locs-locs[c])**2, axis=1))
    w = np.linalg.norm(locs-locs[c], 2, axis=1)
    #arclen = 2*np.arcsin(w/2)
    arclen = np.arcsin(w/2.*np.sqrt(4.-w*w))
    phi_re = locs[:,0]-locs[c][0]
    phi_im = locs[:,1]-locs[c][1]
    #print(type(phi_re), phi_re)
    #print(type(phi_im), phi_im)
    tmp = phi_re + 1j*phi_im
    #tmp = map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1])
    #print(type(tmp), tmp)
    phi = np.angle(tmp)
    #phi = np.angle(map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1]))
    X = arclen*np.real(np.exp(1j*phi))
    Y = arclen*np.imag(np.exp(1j*phi))
    r = max([max(X),max(Y)])
    Xi = np.linspace(-r,r,n_grid)
    Yi = np.linspace(-r,r,n_grid)
    data_ip = griddata((X, Y), data, (Xi[None,:], Yi[:,None]), method='cubic')
    return data_ip


def eeg2map(data):
    """Interpolate and normalize EEG topography, ignoring nan values

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: interger, interpolate to n_grid x n_grid array, default=64
    Returns:
        top_norm: normalized topography, n_grid x n_grid
    """
    n_grid = 64
    top = topo(data, n_grid)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm




def clustering(data, fs, chs, locs, mode, n_clusters, n_win=3, \
               interpol=True, doplot=False):
    """EEG microstate clustering algorithms.

    Args:
        data: numpy.array (n_t, n_ch)
        fs  : sampling frequency [Hz]
        chs : list of channel name strings
        locs: numpy.array (n_ch, 2) of electrode locations (x, y)
        mode: clustering algorithm
        n_clusters: number of clusters
        n_win: smoothing window size 2*n_win+1 [t-n_win:t+n_win]
        doplot: boolean
    Returns:
        maps: n_maps x n_channels NumPy array
        L: microstate sequence (integers)
        gfp_peaks: locations of local GFP maxima
        gev: global explained variance
    """
    #print("[+] Clustering algorithm: {:s}".format(mode))

    # --- number of EEG channels ---
    n_ch = data.shape[1]
    #print("[+] EEG channels: n_ch = {:d}".format(n_ch))

    # --- normalized data ---
    data_norm = data - data.mean(axis=1, keepdims=True)
    data_norm /= data_norm.std(axis=1, keepdims=True)

    # --- GFP peaks ---
    gfp = np.nanstd(data, axis=1)
    gfp2 = np.sum(gfp**2) # normalizing constant
    gfp_peaks = locmax(gfp)
    data_cluster = data[gfp_peaks,:]
    #data_cluster = data_cluster[:100,:]
    data_cluster_norm = data_cluster - data_cluster.mean(axis=1, keepdims=True)
    data_cluster_norm /= data_cluster_norm.std(axis=1, keepdims=True)
    print("[+] Data format for clustering [GFP peaks, channels]: {:d} x {:d}"\
         .format(data_cluster.shape[0], data_cluster.shape[1]))

    start = time.time()

    if (mode == "aahc"):
        print("\n[+] Clustering algorithm: AAHC.")
        maps = aahc(data, n_clusters, doplot=False)

    if (mode == "kmeans"):
        print("\n[+] Clustering algorithm: mod. K-MEANS.")
        maps = kmeans(data, n_maps=n_clusters, n_runs=500)

    if (mode == "kmedoids"):
        print("\n[+] Clustering algorithm: K-MEDOIDS.")
        C = np.corrcoef(data_cluster_norm)
        C = C**2 # ignore EEG polarity
        kmed_maps = kmedoids(S=C, K=n_clusters, nruns=10, maxits=500)
        maps = [int(data_cluster[kmed_maps[k],:]) for k in range(n_clusters)]
        maps = np.array(maps)
        del C, kmed_maps

    if (mode == "pca"):
        print("\n[+] Clustering algorithm: PCA.")
        params = {
            "n_components": n_clusters,
            "copy": True,
            "whiten": True,
            "svd_solver": "auto",
        }
        pca = PCA(**params) # SKLEARN
        pca.fit(data_cluster_norm)
        maps = np.array([pca.components_[k,:] for k in range(n_clusters)])
        '''
        print("PCA explained variance: ", str(pca.explained_variance_ratio_))
        print("PCA total explained variance: ", \
              str(np.sum(pca.explained_variance_ratio_)))
        '''
        del pca, params

        ''' SKLEARN:
        params = {
            "n_components": n_clusters,
            "algorithm": "randomized",
            "n_iter": 5,
            "random_state": None,
            "tol": 0.0,
        }
        svd = TruncatedSVD(**params)
        svd.fit(data_cluster_norm)
        maps = svd.components_
        print("explained variance (%): ")
        print(explained_variance_ratio_)
        #print("explained variance: ")
        print(explained_variance_)
        del svd, params
        '''

    if (mode == "ica"):
        print("\n[+] Clustering algorithm: Fast-ICA.")
        #''' Fast-ICA: algorithm= parallel;deflation, fun=logcosh;exp;cube
        params = {
            "n_components": n_clusters,
            "algorithm": "parallel",
            "whiten": True,
            "fun": "exp",
            "max_iter": 200,
        }
        ica = FastICA(**params) # SKLEARN
        S_ = ica.fit_transform(data_cluster_norm)  # reconstructed signals
        A_ = ica.mixing_  # estimated mixing matrix
        IC_ = ica.components_
        print("data: " + str(data_cluster_norm.shape))
        print("ICs: " + str(IC_.shape))
        print("mixing matrix: " + str(A_.shape))
        maps = np.array([ica.components_[k,:] for k in range(n_clusters)])
        del ica, params

    end = time.time()
    delta_t = end - start
    print(f"[+] Computation time: {delta_t:.2f} sec")

    # --- microstate sequence ---
    print("\n[+] Microstate back-fitting:")
    print("data_norm: ", data_norm.shape)
    print("data_cluster_norm: ", data_cluster_norm.shape)
    print("maps: ", maps.shape)

    if interpol:
        C = np.dot(data_cluster_norm, maps.T)/n_ch
        L_gfp = np.argmax(C**2, axis=1) # microstates at GFP peak
        del C
        n_t = data_norm.shape[0]
        L = np.zeros(n_t)
        for t in range(n_t):
            if t in gfp_peaks:
                i = gfp_peaks.tolist().index(t)
                L[t] = L_gfp[i]
            else:
                i = np.argmin(np.abs(t-gfp_peaks))
                L[t] = L_gfp[i]
        L = L.astype('int')
    else:
        C = np.dot(data_norm, maps.T)/n_ch
        L = np.argmax(C**2, axis=1)
        del C

    # visualize microstate sequence
    if False:
        t_ = np.arange(n_t)
        fig, ax = plt.subplots(2, 1, figsize=(15,8), sharex=True)
        ax[0].plot(t_, gfp, '-k', lw=2)
        for p in gfp_peaks:
            ax[0].axvline(t_[p], c='k', lw=0.5, alpha=0.3)
        ax[0].plot(t_[gfp_peaks], gfp[gfp_peaks], 'or', ms=10)
        ax[1].plot(L)
        plt.show()

    ''' --- temporal smoothing ---
    L2 = np.copy(L)
    for i in range(n_win, len(L)-n_win):
        s = np.array([np.sum(L[i-n_win:i+n_win]==j) for j in range(n_clusters)])
        L2[i] = np.argmax(s)
    L = L2.copy()
    del L2
    '''

    # --- GEV ---
    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    maps_norm /= maps_norm.std(axis=1, keepdims=True)

    # --- correlation data, maps ---
    C = np.dot(data_norm, maps_norm.T)/n_ch
    #print("C.shape: " + str(C.shape))
    #print("C.min: {C.min():.2f}   Cmax: {C.max():.2f}")

    # --- GEV_k & GEV ---
    gev = np.zeros(n_clusters)
    for k in range(n_clusters):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    print(f"\n[+] Global explained variance GEV = {gev.sum():.3f}")
    for k in range(n_clusters):
        print(f"GEV_{k:d}: {gev[k]:.3f}")

    if doplot:
        #plt.ion()
        # matplotlib's perceptually uniform sequential colormaps:
        # magma, inferno, plasma, viridis
        #cm = plt.cm.magma
        cm = plt.cm.seismic
        fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
        for imap in range(n_clusters):
            axarr[imap].imshow(eeg2map(maps[imap, :]), cmap=cm, origin='lower')
            axarr[imap].set_xticks([])
            axarr[imap].set_xticklabels([])
            axarr[imap].set_yticks([])
            axarr[imap].set_yticklabels([])
        title = f"Microstate maps ({mode.upper():s})"
        axarr[0].set_title(title, fontsize=16, fontweight="bold")
 

        # dummy function, callback from TextBox, does nothing
        def f_dummy(text): pass

        axbox = plt.axes([0.1, 0.05, 0.1, 0.1]) #  l, b, w, h
        text_box = TextBox(axbox, 'Ordering: ', initial="[0, 1, 2, 3]")
        text_box.on_submit(f_dummy)
        plt.show()
        order_str = text_box.text
        
        #plt.ion()
        #plt.show()
        #plt.pause(0.001)

        #txt = input("enter something: ")
        #print("txt: ", txt)
        #plt.draw()
        #plt.ioff()

        # --- assign map labels manually ---
        "while ((n_iter==0) ):"
            # (step 3) microstate sequence (= current cluster assignment)
        "C = np.dot(V, maps.T)"
        "C /= (n_ch*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))"
        "L = np.argmax(C**2, axis=1)"
        #order_str = raw_input("\n\t\tAssign map labels (e.g. 0, 2, 1, 3): ")
        #order_str = input("\n\t\tAssign map labels (e.g. 0, 2, 1, 3): ")
        #plt.ioff()
        order_str = order_str.replace("[", "")
        order_str = order_str.replace("]", "")
        order_str = order_str.replace(",", "")
        order_str = order_str.replace(" ", "")
        if (len(order_str) != n_clusters):
            if (len(order_str)==0):
                print("Empty input string.")
            else:
                input_str = ", ".join(order_str)
                print(f"Parsed manual input: {input_str:s}")
                print("Number of labels does not equal number of clusters.")
            print("Continue using the original assignment...\n")
        else:
            order = np.zeros(n_clusters, dtype=int)
            for i, s in enumerate(order_str):
                order[i] = int(s)
            print("Re-ordered labels: {:s}".format(", ".join(order_str)))
            # re-order return variables
            maps = maps[order,:]
            for i in range(len(L)):
                L[i] = order[L[i]]
            gev = gev[order]
            # Figure
            

    return maps, L, gfp_peaks, gev


def aahc(data, N_clusters, doplot=False):
    """Atomize and Agglomerative Hierarchical Clustering Algorithm
    AAHC (Murray et al., Brain Topography, 2008)

    Args:
        data: EEG data to cluster, numpy.array (n_samples, n_channels)
        N_clusters: desired number of clusters
        doplot: boolean, plot maps
    Returns:
        maps: n_maps x n_channels (numpy.array)
    """

    def extract_row(A, k):
        v = A[k,:]
        A_ = np.vstack((A[:k,:],A[k+1:,:]))
        return A_, v

    def extract_item(A, k):
        a = A[k]
        A_ = A[:k] + A[k+1:]
        return A_, a

    #print("\n\t--- AAHC ---")
    nt, nch = data.shape

    # --- get GFP peaks ---
    gfp = data.std(axis=1)
    gfp_peaks = locmax(gfp)
    #gfp_peaks = gfp_peaks[:100]
    #n_gfp = gfp_peaks.shape[0]
    gfp2 = np.sum(gfp**2) # normalizing constant in GEV

    # --- initialize clusters ---
    maps = data[gfp_peaks,:]
    # --- store original gfp peaks and indices ---
    cluster_data = data[gfp_peaks,:]
    #n_maps = n_gfp
    n_maps = maps.shape[0]
    print(f"[+] Initial number of clusters: {n_maps:d}\n")

    # --- cluster indices w.r.t. original size, normalized GFP peak data ---
    Ci = [[k] for k in range(n_maps)]

    # --- main loop: atomize + agglomerate ---
    while (n_maps > N_clusters):
        blank_ = 80*" "
        print(f"\r{blank_:s}\r\t\tAAHC > n: {n_maps:d} => {n_maps-1:d}", end="")
        #stdout.write(s); stdout.flush()
        #print("\n\tAAHC > n: {:d} => {:d}".format(n_maps, n_maps-1))

        # --- correlations of the data sequence with each cluster ---
        m_x, s_x = data.mean(axis=1, keepdims=True), data.std(axis=1)
        m_y, s_y = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
        s_xy = 1.*nch*np.outer(s_x, s_y)
        C = np.dot(data-m_x, np.transpose(maps-m_y)) / s_xy

        # --- microstate sequence, ignore polarity ---
        L = np.argmax(C**2, axis=1)

        # --- GEV (global explained variance) of cluster k ---
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2

        # --- merge cluster with the minimum GEV ---
        imin = np.argmin(gev)
        #print("\tre-cluster: {:d}".format(imin))

        # --- N => N-1 ---
        maps, _ = extract_row(maps, imin)
        Ci, reC = extract_item(Ci, imin)
        re_cluster = []  # indices of updated clusters
        #C_sgn = np.zeros(nt)
        for k in reC:  # map index to re-assign
            c = cluster_data[k,:]
            m_x, s_x = maps.mean(axis=1, keepdims=True), maps.std(axis=1)
            m_y, s_y = c.mean(), c.std()
            s_xy = 1.*nch*s_x*s_y
            C = np.dot(maps-m_x, c-m_y)/s_xy
            inew = np.argmax(C**2) # ignore polarity
            #C_sgn[k] = C[inew]
            re_cluster.append(inew)
            Ci[inew].append(k)
        n_maps = len(Ci)

        # --- update clusters ---
        re_cluster = list(set(re_cluster)) # unique list of updated clusters

        ''' re-clustering by modified mean
        for i in re_cluster:
            idx = Ci[i]
            c = np.zeros(nch) # new cluster average
            # add to new cluster, polarity according to corr. sign
            for k in idx:
                if (C_sgn[k] >= 0):
                    c += cluster_data[k,:]
                else:
                    c -= cluster_data[k,:]
            c /= len(idx)
            maps[i] = c
            #maps[i] = (c-np.mean(c))/np.std(c) # normalize the new cluster
        del C_sgn
        '''

        # re-clustering by eigenvector method
        for i in re_cluster:
            idx = Ci[i]
            Vt = cluster_data[idx,:]
            Sk = np.dot(Vt.T, Vt)
            evals, evecs = np.linalg.eig(Sk)
            c = evecs[:, np.argmax(np.abs(evals))]
            c = np.real(c)
            maps[i] = c/np.sqrt(np.sum(c**2))

    print()
    return maps


def kmeans(data, n_maps, n_runs=10, maxerr=1e-6, maxiter=500):
    """Modified K-means clustering as detailed in:
    [1] Pascual-Marqui et al., IEEE TBME (1995) 82(7):658--665
    [2] Murray et al., Brain Topography(2008) 20:289--268.
    Variables named as in [1], step numbering as in Table I.

    Args:
        data: numpy.array, size = number of EEG channels
        n_maps: number of microstate maps
        n_runs: number of K-means runs (optional)
        maxerr: maximum error for convergence (optional)
        maxiter: maximum number of iterations (optional)
    Returns:
        maps: microstate maps (number of maps x number of channels)
        L: sequence of microstate labels
        gfp_peaks: indices of local GFP maxima
        gev: global explained variance (0..1)
        cv: value of the cross-validation criterion
    """
    n_t = data.shape[0]
    n_ch = data.shape[1]
    data = data - data.mean(axis=1, keepdims=True)

    # GFP peaks
    gfp = np.std(data, axis=1)
    gfp_peaks = locmax(gfp)
    gfp_values = gfp[gfp_peaks]
    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = gfp_peaks.shape[0]

    # clustering of GFP peak maps only
    V = data[gfp_peaks, :]
    sumV2 = np.sum(V**2)

    # store results for each k-means run
    cv_list =   []  # cross-validation criterion for each k-means run
    gev_list =  []  # GEV of each map for each k-means run
    gevT_list = []  # total GEV values for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list =    []  # microstate label sequence for each k-means run
    for run in range(n_runs):
        # initialize random cluster centroids (indices w.r.t. n_gfp)
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
            # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch*np.outer(gfp[gfp_peaks], np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 8)
            for k in range(n_maps):
                Vt = V[L==k, :]
                # (step 8a)
                Sk = np.dot(Vt.T, Vt)
                # (step 8b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                v = v.real
                maps[k, :] = v/np.sqrt(np.sum(v**2))
            # (step 5)
            var1 = var0
            var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
            var0 /= (n_gfp*(n_ch-1))
            n_iter += 1
        if (n_iter < maxiter):
            print((f"\tK-means run {run+1:d}/{n_runs:d} converged after "
                   f"{n_iter:d} iterations."))
        else:
            print((f"\tK-means run {run+1:d}/{n_runs:d} did NOT converge "
                   f"after {maxiter:d} iterations."))

        # CROSS-VALIDATION criterion for this run (step 8)
        C_ = np.dot(data, maps.T)
        C_ /= (n_ch*np.outer(gfp, np.std(maps, axis=1)))
        L_ = np.argmax(C_**2, axis=1)
        var = np.sum(data**2) - np.sum(np.sum(maps[L_, :]*data, axis=1)**2)
        var /= (n_t*(n_ch-1))
        cv = var * (n_ch-1)**2/(n_ch-n_maps-1.)**2

        # GEV (global explained variance) of cluster k
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp_values[r]**2 * C[r,k]**2)/gfp2
        gev_total = np.sum(gev)

        # store
        cv_list.append(cv)
        gev_list.append(gev)
        gevT_list.append(gev_total)
        maps_list.append(maps)
        L_list.append(L_)

    # select best run
    k_opt = np.argmin(cv_list)
    #k_opt = np.argmax(gevT_list)
    maps = maps_list[k_opt]
    # ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
    gev = gev_list[k_opt]
    L_ = L_list[k_opt]

    return maps


def kmedoids(S, K, nruns, maxits):
    """Octave/Matlab: Copyright Brendan J. Frey and Delbert Dueck, Aug 2006
    http://www.psi.toronto.edu/~frey/apm/kcc.m
    Simplified by Kevin Murphy
    Python 2.7.x - FvW, 02/2018

    Args:
        filename: full path to the '.xyz' file
    Returns:
        locs: n_channels x 3 (numpy.array)
    """
    n = S.shape[0]
    dpsim = np.zeros((maxits,nruns))
    idx = np.zeros((n,nruns))
    for rep in range(nruns):
        tmp = np.random.permutation(range(n))
        mu = tmp[:K]
        t = 0
        done = (t==maxits)
        while ( not done ):
            t += 1
            muold = mu
            dpsim[t,rep] = 0
            # Find class assignments
            cl = np.argmax(S[:,mu], axis=1) # max pos. of each row
            # Set assignments of exemplars to themselves
            cl[mu] = range(K)
            for j in range(K): # For each class, find new exemplar
                I = np.where(cl==j)[0]
                S_I_rowsum = np.sum(S[I][:,I],axis=0)
                Scl = max(S_I_rowsum)
                ii = np.argmax(S_I_rowsum)
                dpsim[t,rep] = dpsim[t,rep] + Scl
                mu[j] = I[ii]
            if all(muold==mu) | (t==maxits):
                done = 1
        idx[:,rep] = mu[cl]
        dpsim[t+1:,rep] = dpsim[t,rep]
    return np.unique(idx)


def p_empirical(data, n_clusters):
    """Empirical symbol distribution

    Args:
        data: numpy.array, size = length of microstate sequence
        n_clusters: number of microstate clusters
    Returns:
        p: empirical distribution
    """
    p = np.zeros(n_clusters)
    n = len(data)
    for i in range(n):
        p[data[i]] += 1.0
    p /= n
    return p


def T_empirical(data, n_clusters):
    """Empirical transition matrix

    Args:
        data: numpy.array, size = length of microstate sequence
        n_clusters: number of microstate clusters
    Returns:
        T: empirical transition matrix
    """
    T = np.zeros((n_clusters, n_clusters))
    n = len(data)
    for i in range(n-1):
        T[data[i], data[i+1]] += 1.0
    p_row = np.sum(T, axis=1)
    for i in range(n_clusters):
        if ( p_row[i] != 0.0 ):
            for j in range(n_clusters):
                T[i,j] /= p_row[i]  # normalize row sums to 1.0
    return T


def p_equilibrium(T):
    '''
    get equilibrium distribution from transition matrix:
    lambda = 1 - (left) eigenvector
    '''
    evals, evecs = np.linalg.eig(T.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0] # locate max eigenval.
    p_eq = np.abs(evecs[:,i]) # make eigenvec. to max. eigenval. non-negative
    p_eq /= p_eq.sum() # normalized eigenvec. to max. eigenval.
    return p_eq # stationary distribution


def max_entropy(n_clusters):
    """Maximum Shannon entropy of a sequence with n_clusters

    Args:
        n_clusters: number of microstate clusters
    Returns:
        h_max: maximum Shannon entropy
    """
    h_max = np.log(float(n_clusters))
    return h_max


def shuffle(data):
    """Randomly shuffled copy of data (i.i.d. surrogate)

    Args:
        data: numpy array, 1D
    Returns:
        data_copy: shuffled data copy
    """
    data_c = data.copy()
    np.random.shuffle(data_c)
    return data_c


def surrogate_mc(p, T, n_clusters, n):
    """Surrogate Markov chain with symbol distribution p and
    transition matrix T

    Args:
        p: empirical symbol distribution
        T: empirical transition matrix
        n_clusters: number of clusters/symbols
        n: length of surrogate microstate sequence
    Returns:
        mc: surrogate Markov chain
    """

    # NumPy vectorized code
    psum = np.cumsum(p)
    Tsum = np.cumsum(T, axis=1)
    # initial state according to p:
    mc = [np.min(np.argwhere(np.random.rand() < psum))]
    # next state according to T:
    for i in range(1, n):
        mc.append(np.min(np.argwhere(np.random.rand() < Tsum[mc[i-1]])))

    ''' alternative implementation using loops
    r = np.random.rand() # ~U[0,1], random threshold
    s = 0.
    y = p[s]
    while (y < r):
        s += 1.
        y += p[s]
    mc = [s] # initial state according to p

    # iterate ...
    for i in xrange(1,n):
        r = np.random.rand() # ~U[0,1], random threshold
        s = mc[i-1] # currrent state
        t = 0. # trial state
        y = T[s][t] # transition rate to trial state
        while ( y < r ):
            t += 1. # next trial state
            y += T[s][t]
        mc.append(t) # store current state
    '''

    return np.array(mc)


def mutinf(x, ns, lmax):
    """Time-lagged mutual information of symbolic sequence x with
    ns different symbols, up to maximum lag lmax.
    *** Symbols must be 0, 1, 2, ... to use as indices directly! ***

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        lmax: maximum time lag
    Returns:
        mi: time-lagged mutual information
    """

    n = len(x)
    mi = np.zeros(lmax)
    for l in range(lmax):
        if ((l+1)%10 == 0):
            print(f"mutual information lag: {l+1:d}\r", end="")
            #sys.stdout.write(s)
            #sys.stdout.flush()
        nmax = n-l
        h1 = H_1(x[:nmax], ns)
        h2 = H_1(x[l:l+nmax], ns)
        h12 = H_2(x[:nmax], x[l:l+nmax], ns)
        mi[l] = h1 + h2 - h12
    print("")
    return mi


def mutinf_i(x, ns, lmax):
    """Time-lagged mutual information for each symbol of the symbolic
    sequence x with  ns different symbols, up to maximum lag lmax.
    *** Symbols must be 0, 1, 2, ... to use as indices directly! ***

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        lmax: maximum time lag
    Returns:
        mi: time-lagged mutual informations for each symbol, ns x lmax
    """

    n = len(x)
    mi = np.zeros((ns,lmax))
    for l in range(lmax):
        if (l%10 == 0):
            print(f"\r\tmutual information lag: {l:d}\r", end="")
            #sys.stdout.write(s)
            #sys.stdout.flush()
        nmax = n - l

        # compute distributions
        p1 = np.zeros(ns)
        p2 = np.zeros(ns)
        p12 = np.zeros((ns,ns))
        for i in range(nmax):
            i1 = int(x[i])
            i2 = int(x[i+l])
            p1[i1] += 1.0 # p( X_t = i1 )
            p2[i2] += 1.0 # p( X_t+l = i2 )
            p12[i1,i2] += 1.0 # p( X_t+l=i2 , X_t=i1 )
        p1 /= nmax
        p2 /= nmax

        # normalize the transition matrix p( X_t+l=i2 | X_t=i1 )
        rowsum = np.sum(p12, axis=1)
        for i, j in np.ndindex(p12.shape):
            p12[i,j] /= rowsum[i]

        # compute entropies
        H2 = np.sum(p2[p2>0] * np.log(p2[p2>0]))
        for i in range(ns):
            H12 = 0.0
            for j in range(ns):
                if ( p12[i,j] > 0.0 ):
                    H12 += ( p12[i,j] * np.log( p12[i,j] ) )
            mi[i,l] = -H2 + p1[i] * H12
    return mi


def mutinf_CI(p, T, n, alpha, nrep, lmax):
    """Return an array for the computation of confidence intervals (CI) for
    the time-lagged mutual information of length lmax.
    Null Hypothesis: Markov Chain of length n, equilibrium distribution p,
    transition matrix T, using nrep repetitions."""

    ns = len(p)
    mi_array = np.zeros((nrep,lmax))
    for r in range(nrep):
        print(f"\nsurrogate MC # {r+1:d}/{nrep:d}")
        x_mc = surrogate_mc(p, T, ns, n)
        mi_mc = mutinf(x_mc, ns, lmax)
        #mi_mc = mutinf_cy(X_mc, ns, lmax)
        mi_array[r,:] = mi_mc
    return mi_array


def excess_entropy_rate(x, ns, kmax, doplot=False):
    # y = ax+b: line fit to joint entropy for range of histories k
    # a = entropy rate (slope)
    # b = excess entropy (intersect.)
    h_ = np.zeros(kmax)
    for k in range(kmax): h_[k] = H_k(x, ns, k+1)
    ks = np.arange(1,kmax+1)
    a, b = np.polyfit(ks, h_, 1)
    # --- Figure ---
    if doplot:
        plt.figure(figsize=(6,6))
        plt.plot(ks, h_, '-sk')
        plt.plot(ks, a*ks+b, '-b')
        plt.xlabel("k")
        plt.ylabel("$H_k$")
        plt.title("Entropy rate")
        plt.tight_layout()
        plt.show()
    return (a, b)


def mc_entropy_rate(p, T):
    """Markov chain entropy rate.
    - \sum_i sum_j p_i T_ij log(T_ij)
    """
    h = 0.
    for i, j in np.ndindex(T.shape):
        if (T[i,j] > 0):
            h -= ( p[i]*T[i,j]*np.log(T[i,j]) )
    return h


def aif_peak1(mi, fs, doplot=False):
    '''compute time-lagged mut. inf. (AIF) and 1st peak.'''
    dt = 1000./fs # sampling interval [ms]
    mi_filt = np.convolve(mi, np.ones(3)/3., mode='same')
    #mi_filt = np.convolve(mi, np.ones(5)/5., mode='same')
    mx0 = 8 # 8
    jmax = mx0 + locmax(mi_filt[mx0:])[0]
    mx_mi = dt*jmax
    if doplot:
        offset = 5
        tmax = 100
        fig = plt.figure(1, figsize=(22,8))
        t = dt*np.arange(tmax)
        plt.plot(t[offset:tmax], mi[offset:tmax], '-ok', label='AIF')
        plt.plot(t[offset:tmax], mi_filt[offset:tmax], '-b', label='smoothed AIF')
        plt.plot(mx_mi, mi[jmax], 'or', markersize=15, label='peak-1')
        plt.xlabel("time lag [ms]")
        plt.ylabel("mut. inf. [bits]")
        plt.legend(loc=0)
        #plt.title("mutual information of map sequence")
        #plt.title(s, fontsize=16, fontweight='bold')
        plt.show()
        input()

    return jmax, mx_mi


def H_1(x, ns):
    """Shannon entropy of the symbolic sequence x with ns symbols.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    n = len(x)
    p = np.zeros(ns) # symbol distribution
    for t in range(n):
        p[x[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_2(x, y, ns):
    """Joint Shannon entropy of the symbolic sequences X, Y with ns symbols.

    Args:
        x, y: symbolic sequences, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        h: Shannon entropy of x
    """

    if (len(x) != len(y)):
        print("H_2 warning: sequences of different lengths, using the shorter...")
    n = min([len(x), len(y)])
    p = np.zeros((ns, ns)) # joint distribution
    for t in range(n):
        p[x[t],y[t]] += 1.0
    p /= n
    h = -np.sum(p[p>0]*np.log(p[p>0]))
    return h


def H_k(x, ns, k):
    """Shannon's joint entropy from x[n+p:n-m]
    x: symbolic time series
    ns: number of symbols
    k: length of k-history
    """

    N = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(N-k): f[tuple(x[t:t+k])] += 1.0
    f /= (N-k) # normalize distribution
    hk = -np.sum(f[f>0]*np.log(f[f>0]))
    #m = np.sum(f>0)
    #hk = hk + (m-1)/(2*N) # Miller-Madow bias correction
    return hk


def testMarkov0(x, ns, alpha, verbose=True):
    """Test zero-order Markovianity of symbolic sequence x with ns symbols.
    Null hypothesis: zero-order MC (iid) <=>
    p(X[t]), p(X[t+1]) independent
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("ZERO-ORDER MARKOVIANITY:")
    n = len(x)
    f_ij = np.zeros((ns,ns))
    f_i = np.zeros(ns)
    f_j = np.zeros(ns)
    # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
    for t in range(n-1):
        i = x[t]
        j = x[t+1]
        f_ij[i,j] += 1.0
        f_i[i] += 1.0
        f_j[j] += 1.0
    T = 0.0 # statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            num_ = n*f_ij[i,j]
            den_ = f_i[i]*f_j[j]
            T += (f_ij[i,j] * np.log(num_/den_))
    T *= 2.0
    df = (ns-1.0) * (ns-1.0)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def testMarkov1(X, ns, alpha, verbose=True):
    """Test first-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("\nFIRST-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijk = np.zeros((ns,ns,ns))
    f_ij = np.zeros((ns,ns))
    f_jk = np.zeros((ns,ns))
    f_j = np.zeros(ns)
    for t in range(n-2):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        f_ijk[i,j,k] += 1.0
        f_ij[i,j] += 1.0
        f_jk[j,k] += 1.0
        f_j[j] += 1.0
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i][j][k]*f_j[j]*f_ij[i][j]*f_jk[j][k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def testMarkov2(X, ns, alpha, verbose=True):
    """Test second-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])
    cf. Kullback, Technometrics (1962), Table 10.2.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("\nSECOND-ORDER MARKOVIANITY:")
    n = len(X)
    f_ijkl = np.zeros((ns,ns,ns,ns))
    f_ijk = np.zeros((ns,ns,ns))
    f_jkl = np.zeros((ns,ns,ns))
    f_jk = np.zeros((ns,ns))
    for t in range(n-3):
        i = X[t]
        j = X[t+1]
        k = X[t+2]
        l = X[t+3]
        f_ijkl[i,j,k,l] += 1.0
        f_ijk[i,j,k] += 1.0
        f_jkl[j,k,l] += 1.0
        f_jk[j,k] += 1.0
    T = 0.0
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        f = f_ijkl[i,j,k,l]*f_ijk[i,j,k]*f_jkl[j,k,l]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijkl[i,j,k,l]*f_jk[j,k]
            den_ = f_ijk[i,j,k]*f_jkl[j,k,l]
            T += (f_ijkl[i,j,k,l]*np.log(num_/den_))
    T *= 2.0
    df = ns*ns*(ns-1)*(ns-1)
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def conditionalHomogeneityTest(X, ns, l, alpha, verbose=True):
    """Test conditional homogeneity of non-overlapping blocks of
    length l of symbolic sequence X with ns symbols
    cf. Kullback, Technometrics (1962), Table 9.1.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        l: split x into non-overlapping blocks of size l
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("\nCONDITIONAL HOMOGENEITY (three-way table):")
    n = len(X)
    r = int(np.floor(float(n)/float(l))) # number of blocks
    nl = r*l
    if verbose:
        print("Split data in r = {:d} blocks of length {:d}.".format(r,l))
    f_ijk = np.zeros((r,ns,ns))
    f_ij = np.zeros((r,ns))
    f_jk = np.zeros((ns,ns))
    f_i = np.zeros(r)
    f_j = np.zeros(ns)

    # calculate f_ijk (time / block dep. transition matrix)
    for i in  range(r): # block index
        for ii in range(l-1): # pos. inside the current block
            j = X[i*l + ii]
            k = X[i*l + ii + 1]
            f_ijk[i,j,k] += 1.0
            f_ij[i,j] += 1.0
            f_jk[j,k] += 1.0
            f_i[i] += 1.0
            f_j[j] += 1.0

    # conditional homogeneity (Markovianity stationarity)
    T = 0.0
    for i, j, k in np.ndindex(f_ijk.shape):
        # conditional homogeneity
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            T += (f_ijk[i,j,k]*np.log(num_/den_))
    T *= 2.0
    df = (r-1)*(ns-1)*ns
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def symmetryTest(X, ns, alpha, verbose=True):
    """Test symmetry of the transition matrix of symbolic sequence X with
    ns symbols
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        alpha: significance level
    Returns:
        p: p-value of the Chi2 test for independence
    """

    if verbose:
        print("\nSYMMETRY:")
    n = len(X)
    f_ij = np.zeros((ns,ns))
    for t in range(n-1):
        i = X[t]
        j = X[t+1]
        f_ij[i,j] += 1.0
    T = 0.0
    for i, j in np.ndindex(f_ij.shape):
        if (i != j):
            f = f_ij[i,j]*f_ij[j,i]
            if (f > 0):
                num_ = 2*f_ij[i,j]
                den_ = f_ij[i,j]+f_ij[j,i]
                T += (f_ij[i,j]*np.log(num_/den_))
    T *= 2.0
    df = ns*(ns-1)/2
    #p = chi2test(T, df, alpha)
    p = chi2.sf(T, df, loc=0, scale=1)
    if verbose:
        print(f"p: {p:.2e} | t: {T:.3f} | df: {df:.1f}")
    return p


def geoTest(X, ns, dt, alpha, verbose=True):
    """Test the geometric distribution of lifetime distributions.

    Args:
        X: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
        dt: time step in [ms]
        alpha: significance level
    Returns:
        p_values: for each symbol
    """

    if verbose:
        print("\nGEOMETRIC DISTRIBUTION of lifetimes:\n")
    tauDist = lifetimes(X, ns)
    T = T_empirical(X, ns)
    p_values = np.zeros(ns)
    for s in range(ns): # test for each symbol
        if verbose:
            print(f"\nTesting the distribution of symbol # {s:d}")
        # m = max_tau:
        m = len(tauDist[s])
        # observed lifetime distribution:
        q_obs = np.zeros(m)
        # theoretical lifetime distribution:
        q_exp = np.zeros(m)
        for j in range(m):
            # observed frequency of lifetime j+1 for state s
            q_obs[j] = tauDist[s][j]
            # expected frequency
            q_exp[j] = (1.0-T[s][s]) * (T[s][s]**j)
        q_exp *= sum(q_obs)

        t = 0.0 # chi2 statistic
        for j in range(m):
            if ((q_obs[j] > 0) & (q_exp[j] > 0)):
                t += (q_obs[j]*np.log(q_obs[j]/q_exp[j]))
        t *= 2.0
        df = m-1
        #p0 = chi2test(t, df, alpha)
        p0 = chi2.sf(t, df, loc=0, scale=1)
        if verbose:
            print(f"p: {p0:.2e} | t: {t:.3f} | df: {df:.1f}")
        p_values[s] = p0

        g1, p1, dof1, expctd1 = chi2_contingency(np.vstack((q_obs,q_exp)), \
                                                 lambda_='log-likelihood')
        if verbose:
            print((f"G-test (log-likelihood) p: {p1:.2e}, g: {g1:.3f}, "
                   f"df: {dof1:.1f}"))

        # Pearson's Chi2 test
        g2, p2, dof2, expctd2 = chi2_contingency( np.vstack((q_obs,q_exp)) )
        if verbose:
            print((f"G-test (Pearson Chi2) p: {p2:.2e}, g: {g2:.3f}, "
                   f"df: {dof2:.1f}"))

        p_ = tauDist[s]/np.sum(tauDist[s])
        tau_mean = 0.0
        tau_max = len(tauDist[s])*dt
        for j in range(len(p_)): tau_mean += (p_[j] * j)
        tau_mean *= dt
        if verbose:
            pass
            #print("\t\tmean dwell time: {:.2f} [ms]".format(tau_mean))
        if verbose:
            pass
            #print("\t\tmax. dwell time: {:.2f} [ms]\n\n".format(tau_max))
    return p_values


def lifetimes(X, ns):
    """Compute the lifetime distributions for each symbol
    in a symbolic sequence X with ns symbols.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        ns: number of symbols
    Returns:
        tauDist: list of lists, lifetime distributions for each symbol
    """

    n = len(X)
    tauList = [[] for i in range(ns)] # unsorted lifetimes, [[],[],[],[]]
    i = 0 # current index
    s = X[i] # current symbol
    tau = 1.0 # current lifetime
    while (i < n-1):
        i += 1
        if ( X[i] == s ):
            tau += 1.0
        else:
            tauList[int(s)].append(tau)
            s = X[i]
            tau = 1.0
    tauList[int(s)].append(tau) # last state
    # find max lifetime for each symbol
    tau_max = [max(L) for L in tauList]
    #print( "max lifetime for each symbol : " + str(tau_max) )
    tauDist = [] # empty histograms for each symbol
    for i in range(ns): tauDist.append( [0.]*int(tau_max[i]) )
    # lifetime distributions
    for s in range(ns):
        for j in range(len(tauList[s])):
            tau = tauList[s][j]
            tauDist[s][int(tau)-1] += 1.0
    return tauDist


def mixing_time(X, ns):
    """
    Relaxation time, inverse of spectral gap
    Arguments:
        X: microstate label sequence
        ns: number of unique labels
    """
    T_hat = T_empirical(X,ns) # transition matrix
    ev = np.linalg.eigvals(T_hat)
    #ev = np.real_if_close(ev)
    ev = np.real(ev)
    ev.sort() # ascending
    ev2 = np.flipud(ev) # descending
    #print("ordered eigenvalues: {:s}".format(str(ev2)))
    sg = ev2[0] - ev2[1] # spectral gap
    T_mix = 1.0/sg # mixing time
    return T_mix


def multiple_comparisons(p_values, method):
    """Apply multiple comparisons correction code using the statsmodels package.
    Input: array p-values.
    'b': 'Bonferroni'
    's': 'Sidak'
    'h': 'Holm'
    'hs': 'Holm-Sidak'
    'sh': 'Simes-Hochberg'
    'ho': 'Hommel'
    'fdr_bh': 'FDR Benjamini-Hochberg'
    'fdr_by': 'FDR Benjamini-Yekutieli'
    'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg'
    'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli'
    'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'

    Args:
        p_values: uncorrected p-values
        method: string, one of the methods given above
    Returns:
        p_values_corrected: corrected p-values
    """
    reject, pvals_corr, alphacSidak, alphacBonf = multipletests(p_values, \
                                                                method=method)
    return pvals_corr


def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """

    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


def locmin(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """

    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == +2)[0] # indices of local max.
    return m


def locmax2d(x1):
    """Get local maxima of 2D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """
    # x=topo(x)
    x=np.zeros((5,5))
    x[0][0]=x1[0]
    x[0][1]=x1[0]
    x[0][2]=(x1[0]+x1[1])/2
    x[0][3]=x1[1]
    x[0][4]=x1[1]
    x[1][0]=x1[2]
    x[1][1]=x1[3]
    x[1][2]=x1[4]
    x[1][3]=x1[5]
    x[1][4]=x1[6]
    x[2][0]=x1[7]
    x[2][1]=x1[8]
    x[2][2]=x1[9]
    x[2][3]=x1[10]
    x[2][4]=x1[11]
    x[3][0]=x1[12]
    x[3][1]=x1[13]
    x[3][2]=x1[14]
    x[3][3]=x1[15]
    x[3][4]=x1[16]
    x[4][0]=x1[17]
    x[4][1]=x1[17]
    x[4][2]=(x1[17]+x1[18])/2
    x[4][3]=x1[18]
    x[4][4]=x1[18]
    
    
    r=x.shape[0]
    c=x.shape[1]
    
    m=np.zeros((r,c))
    maxi=float(-10000)
    for i in range(1,r-1):
        for j in range(1,c-1):
            maxip=[]    
            for i1 in range(i-1,i+2):
                for j1 in range(j-1,j+2):
                    if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                        a=0
                    else:
                        if(maxi<float(x[i1][j1])):
                           maxi=float(x[i1][j1])
                    
            # maxi=np.max(x[i-1][j-1],x[i][j-1],x[i+1][j-1],x[i-1][j],x[i+1][j],x[i-1][j+1],x[i][j],x[i+1][j-1])
            if(x[i][j]==maxi):
                m[i][j]=1
            
    return m

def findstr(s, L):
    """Find string in list of strings, returns indices.

    Args:
        s: query string
        L: list of strings to search
    Returns:
        x: list of indices where s is found in L
    """

    x = [i for i, l in enumerate(L) if (l==s)]
    return x


def print_matrix(T):
    """Console-friendly output of the matrix T.

    Args:
        T: matrix to print
    """

    for i, j in np.ndindex(T.shape):
        if (j == 0):
            print("|{:.3f}".format(T[i,j]), end='')
        elif (j == T.shape[1]-1):
            print("{:.3f}|\n".format(T[i,j]), end='')
        else:
            #print "{:.3f}".format(T[i,j]), # Python-2
            print("{:.3f}".format(T[i,j]), end='') # Python-3


# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:23:83 2021

@author: win
"""

filename=[]
filename1 = ["data_Subject00_1_1.edf","data_Subject00_1_2.edf","data_Subject00_1_3.edf","data_Subject00_1_4.edf","data_Subject00_1_5.edf","data_Subject00_1_6.edf","data_Subject00_1_7.edf","data_Subject00_1_8.edf","data_Subject00_1_9.edf","data_Subject00_1_10.edf","data_Subject00_1_11.edf","data_Subject00_1_12.edf","data_Subject00_1_13.edf","data_Subject00_1_14.edf","data_Subject00_1_15.edf","data_Subject00_1_16.edf","data_Subject00_1_18.edf","data_Subject00_1_19.edf","data_Subject00_1_20.edf","data_Subject00_1_21.edf","data_Subject00_1_22.edf","data_Subject00_1_23.edf"]
filename2 = ["data_Subject01_1_1.edf","data_Subject01_1_2.edf","data_Subject01_1_3.edf","data_Subject01_1_4.edf","data_Subject01_1_5.edf","data_Subject01_1_6.edf","data_Subject01_1_7.edf","data_Subject01_1_8.edf","data_Subject01_1_9.edf","data_Subject01_1_10.edf","data_Subject01_1_11.edf","data_Subject01_1_12.edf","data_Subject01_1_13.edf","data_Subject01_1_14.edf","data_Subject01_1_15.edf","data_Subject01_1_16.edf","data_Subject01_1_18.edf","data_Subject01_1_19.edf","data_Subject01_1_20.edf","data_Subject01_1_21.edf","data_Subject01_1_22.edf","data_Subject01_1_23.edf"]
filename3 = ["data_Subject02_1_1.edf","data_Subject02_1_2.edf","data_Subject02_1_3.edf","data_Subject02_1_4.edf","data_Subject02_1_5.edf","data_Subject02_1_6.edf","data_Subject02_1_7.edf","data_Subject02_1_8.edf","data_Subject02_1_9.edf","data_Subject02_1_10.edf","data_Subject02_1_11.edf","data_Subject02_1_12.edf","data_Subject02_1_13.edf","data_Subject02_1_14.edf","data_Subject02_1_15.edf","data_Subject02_1_16.edf","data_Subject02_1_18.edf","data_Subject02_1_19.edf","data_Subject02_1_20.edf","data_Subject02_1_21.edf","data_Subject02_1_22.edf","data_Subject02_1_23.edf"]
filename4 = ["data_Subject03_1_1.edf","data_Subject03_1_2.edf","data_Subject03_1_3.edf","data_Subject03_1_4.edf","data_Subject03_1_5.edf","data_Subject03_1_6.edf","data_Subject03_1_7.edf","data_Subject03_1_8.edf","data_Subject03_1_9.edf","data_Subject03_1_10.edf","data_Subject03_1_11.edf","data_Subject03_1_12.edf","data_Subject03_1_13.edf","data_Subject03_1_14.edf","data_Subject03_1_15.edf","data_Subject03_1_16.edf","data_Subject03_1_18.edf","data_Subject03_1_19.edf","data_Subject03_1_20.edf","data_Subject03_1_21.edf","data_Subject03_1_22.edf","data_Subject03_1_23.edf"]
filename5 = ["data_Subject04_1_1.edf","data_Subject04_1_2.edf","data_Subject04_1_3.edf","data_Subject04_1_4.edf","data_Subject04_1_5.edf","data_Subject04_1_6.edf","data_Subject04_1_7.edf","data_Subject04_1_8.edf","data_Subject04_1_9.edf","data_Subject04_1_10.edf","data_Subject04_1_11.edf","data_Subject04_1_12.edf","data_Subject04_1_13.edf","data_Subject04_1_14.edf","data_Subject04_1_15.edf","data_Subject04_1_16.edf","data_Subject04_1_18.edf","data_Subject04_1_19.edf","data_Subject04_1_20.edf","data_Subject04_1_21.edf","data_Subject04_1_22.edf","data_Subject04_1_23.edf"]
filename6 = ["data_Subject05_1_1.edf","data_Subject05_1_2.edf","data_Subject05_1_3.edf","data_Subject05_1_4.edf","data_Subject05_1_5.edf","data_Subject05_1_6.edf","data_Subject05_1_7.edf","data_Subject05_1_8.edf","data_Subject05_1_9.edf","data_Subject05_1_10.edf","data_Subject05_1_11.edf","data_Subject05_1_12.edf","data_Subject05_1_13.edf","data_Subject05_1_14.edf","data_Subject05_1_15.edf","data_Subject05_1_16.edf","data_Subject05_1_18.edf","data_Subject05_1_19.edf","data_Subject05_1_20.edf","data_Subject05_1_21.edf","data_Subject05_1_22.edf","data_Subject05_1_23.edf"]
filename7 = ["data_Subject06_1_1.edf","data_Subject06_1_2.edf","data_Subject06_1_3.edf","data_Subject06_1_4.edf","data_Subject06_1_5.edf","data_Subject06_1_6.edf","data_Subject06_1_7.edf","data_Subject06_1_8.edf","data_Subject06_1_9.edf","data_Subject06_1_10.edf","data_Subject06_1_11.edf","data_Subject06_1_12.edf","data_Subject06_1_13.edf","data_Subject06_1_14.edf","data_Subject06_1_15.edf","data_Subject06_1_16.edf","data_Subject06_1_18.edf","data_Subject06_1_19.edf","data_Subject06_1_20.edf","data_Subject06_1_21.edf","data_Subject06_1_22.edf","data_Subject06_1_23.edf"]
filename8 = ["data_Subject07_1_1.edf","data_Subject07_1_2.edf","data_Subject07_1_3.edf","data_Subject07_1_4.edf","data_Subject07_1_5.edf","data_Subject07_1_6.edf","data_Subject07_1_7.edf","data_Subject07_1_8.edf","data_Subject07_1_9.edf","data_Subject07_1_10.edf","data_Subject07_1_11.edf","data_Subject07_1_12.edf","data_Subject07_1_13.edf","data_Subject07_1_14.edf","data_Subject07_1_15.edf","data_Subject07_1_16.edf","data_Subject07_1_18.edf","data_Subject07_1_19.edf","data_Subject07_1_20.edf","data_Subject07_1_21.edf","data_Subject07_1_22.edf","data_Subject07_1_23.edf"]
filename9 = ["data_Subject08_1_1.edf","data_Subject08_1_2.edf","data_Subject08_1_3.edf","data_Subject08_1_4.edf","data_Subject08_1_5.edf","data_Subject08_1_6.edf","data_Subject08_1_7.edf","data_Subject08_1_8.edf","data_Subject08_1_9.edf","data_Subject08_1_10.edf","data_Subject08_1_11.edf","data_Subject08_1_12.edf","data_Subject08_1_13.edf","data_Subject08_1_14.edf","data_Subject08_1_15.edf","data_Subject08_1_16.edf","data_Subject08_1_18.edf","data_Subject08_1_19.edf","data_Subject08_1_20.edf","data_Subject08_1_21.edf","data_Subject08_1_22.edf","data_Subject08_1_23.edf"]
filename10= ["data_Subject09_1_1.edf","data_Subject09_1_2.edf","data_Subject09_1_3.edf","data_Subject09_1_4.edf","data_Subject09_1_5.edf","data_Subject09_1_6.edf","data_Subject09_1_7.edf","data_Subject09_1_8.edf","data_Subject09_1_9.edf","data_Subject09_1_10.edf","data_Subject09_1_11.edf","data_Subject09_1_12.edf","data_Subject09_1_13.edf","data_Subject09_1_14.edf","data_Subject09_1_15.edf","data_Subject09_1_16.edf","data_Subject09_1_18.edf","data_Subject09_1_19.edf","data_Subject09_1_20.edf","data_Subject09_1_21.edf","data_Subject09_1_22.edf","data_Subject09_1_23.edf"]
filename11= ["data_Subject10_1_1.edf","data_Subject10_1_2.edf","data_Subject10_1_3.edf","data_Subject10_1_4.edf","data_Subject10_1_5.edf","data_Subject10_1_6.edf","data_Subject10_1_7.edf","data_Subject10_1_8.edf","data_Subject10_1_9.edf","data_Subject10_1_10.edf","data_Subject10_1_11.edf","data_Subject10_1_12.edf","data_Subject10_1_13.edf","data_Subject10_1_14.edf","data_Subject10_1_15.edf","data_Subject10_1_16.edf","data_Subject10_1_18.edf","data_Subject10_1_19.edf","data_Subject10_1_20.edf","data_Subject10_1_21.edf","data_Subject10_1_22.edf","data_Subject10_1_23.edf"]
filename12= ["data_Subject11_1_1.edf","data_Subject11_1_2.edf","data_Subject11_1_3.edf","data_Subject11_1_4.edf","data_Subject11_1_5.edf","data_Subject11_1_6.edf","data_Subject11_1_7.edf","data_Subject11_1_8.edf","data_Subject11_1_9.edf","data_Subject11_1_11.edf","data_Subject11_1_11.edf","data_Subject11_1_11.edf","data_Subject11_1_13.edf","data_Subject11_1_14.edf","data_Subject11_1_15.edf","data_Subject11_1_16.edf","data_Subject11_1_18.edf","data_Subject11_1_19.edf","data_Subject11_1_20.edf","data_Subject11_1_21.edf","data_Subject11_1_22.edf","data_Subject11_1_23.edf"]
filename13= ["data_Subject12_1_1.edf","data_Subject12_1_2.edf","data_Subject12_1_3.edf","data_Subject12_1_4.edf","data_Subject12_1_5.edf","data_Subject12_1_6.edf","data_Subject12_1_7.edf","data_Subject12_1_8.edf","data_Subject12_1_9.edf","data_Subject12_1_10.edf","data_Subject12_1_11.edf","data_Subject12_1_12.edf","data_Subject12_1_13.edf","data_Subject12_1_14.edf","data_Subject12_1_15.edf","data_Subject12_1_16.edf","data_Subject12_1_17.edf","data_Subject12_1_18.edf","data_Subject12_1_19.edf","data_Subject12_1_20.edf","data_Subject12_1_21.edf","data_Subject12_1_22.edf"]
filename14= ["data_Subject13_1_1.edf","data_Subject13_1_2.edf","data_Subject13_1_3.edf","data_Subject13_1_4.edf","data_Subject13_1_5.edf","data_Subject13_1_6.edf","data_Subject13_1_7.edf","data_Subject13_1_8.edf","data_Subject13_1_9.edf","data_Subject13_1_10.edf","data_Subject13_1_11.edf","data_Subject13_1_12.edf","data_Subject13_1_13.edf","data_Subject13_1_14.edf","data_Subject13_1_15.edf","data_Subject13_1_16.edf","data_Subject13_1_17.edf","data_Subject13_1_18.edf","data_Subject13_1_19.edf","data_Subject13_1_20.edf","data_Subject13_1_21.edf","data_Subject13_1_22.edf"]
filename15= ["data_Subject14_1_1.edf","data_Subject14_1_2.edf","data_Subject14_1_3.edf","data_Subject14_1_4.edf","data_Subject14_1_5.edf","data_Subject14_1_6.edf","data_Subject14_1_7.edf","data_Subject14_1_8.edf","data_Subject14_1_9.edf","data_Subject14_1_10.edf","data_Subject14_1_11.edf","data_Subject14_1_12.edf","data_Subject14_1_13.edf","data_Subject14_1_14.edf","data_Subject14_1_15.edf","data_Subject14_1_16.edf","data_Subject14_1_17.edf","data_Subject14_1_18.edf","data_Subject14_1_19.edf","data_Subject14_1_20.edf","data_Subject14_1_21.edf","data_Subject14_1_22.edf"]
filename16= ["data_Subject15_1_1.edf","data_Subject15_1_2.edf","data_Subject15_1_3.edf","data_Subject15_1_4.edf","data_Subject15_1_5.edf","data_Subject15_1_6.edf","data_Subject15_1_7.edf","data_Subject15_1_8.edf","data_Subject15_1_9.edf","data_Subject15_1_10.edf","data_Subject15_1_11.edf","data_Subject15_1_12.edf","data_Subject15_1_13.edf","data_Subject15_1_14.edf","data_Subject15_1_15.edf","data_Subject15_1_16.edf","data_Subject15_1_17.edf","data_Subject15_1_18.edf","data_Subject15_1_19.edf","data_Subject15_1_20.edf","data_Subject15_1_21.edf","data_Subject15_1_22.edf"]
filename17= ["data_Subject16_1_1.edf","data_Subject16_1_2.edf","data_Subject16_1_3.edf","data_Subject16_1_4.edf","data_Subject16_1_5.edf","data_Subject16_1_6.edf","data_Subject16_1_7.edf","data_Subject16_1_8.edf","data_Subject16_1_9.edf","data_Subject16_1_10.edf","data_Subject16_1_11.edf","data_Subject16_1_12.edf","data_Subject16_1_13.edf","data_Subject16_1_14.edf","data_Subject16_1_15.edf","data_Subject16_1_16.edf","data_Subject16_1_17.edf","data_Subject16_1_18.edf","data_Subject16_1_19.edf","data_Subject16_1_20.edf","data_Subject16_1_21.edf","data_Subject16_1_22.edf"]
filename18= ["data_Subject17_1_1.edf","data_Subject17_1_2.edf","data_Subject17_1_3.edf","data_Subject17_1_4.edf","data_Subject17_1_5.edf","data_Subject17_1_6.edf","data_Subject17_1_7.edf","data_Subject17_1_8.edf","data_Subject17_1_9.edf","data_Subject17_1_10.edf","data_Subject17_1_11.edf","data_Subject17_1_12.edf","data_Subject17_1_13.edf","data_Subject17_1_14.edf","data_Subject17_1_15.edf","data_Subject17_1_16.edf","data_Subject17_1_17.edf","data_Subject17_1_18.edf","data_Subject17_1_19.edf","data_Subject17_1_20.edf","data_Subject17_1_21.edf","data_Subject17_1_22.edf"]
filename19= ["data_Subject18_1_1.edf","data_Subject18_1_2.edf","data_Subject18_1_3.edf","data_Subject18_1_4.edf","data_Subject18_1_5.edf","data_Subject18_1_6.edf","data_Subject18_1_7.edf","data_Subject18_1_8.edf","data_Subject18_1_9.edf","data_Subject18_1_10.edf","data_Subject18_1_11.edf","data_Subject18_1_12.edf","data_Subject18_1_13.edf","data_Subject18_1_14.edf","data_Subject18_1_15.edf","data_Subject18_1_16.edf","data_Subject18_1_17.edf","data_Subject18_1_18.edf","data_Subject18_1_19.edf","data_Subject18_1_20.edf","data_Subject18_1_21.edf","data_Subject18_1_22.edf"]
filename20= ["data_Subject19_1_1.edf","data_Subject19_1_2.edf","data_Subject19_1_3.edf","data_Subject19_1_4.edf","data_Subject19_1_5.edf","data_Subject19_1_6.edf","data_Subject19_1_7.edf","data_Subject19_1_8.edf","data_Subject19_1_9.edf","data_Subject19_1_10.edf","data_Subject19_1_11.edf","data_Subject19_1_12.edf","data_Subject19_1_13.edf","data_Subject19_1_14.edf","data_Subject19_1_15.edf","data_Subject19_1_16.edf","data_Subject19_1_17.edf","data_Subject19_1_18.edf","data_Subject19_1_19.edf","data_Subject19_1_20.edf","data_Subject19_1_21.edf","data_Subject19_1_22.edf"]
filename21= ["data_Subject20_1_1.edf","data_Subject20_1_2.edf","data_Subject20_1_3.edf","data_Subject20_1_4.edf","data_Subject20_1_5.edf","data_Subject20_1_6.edf","data_Subject20_1_7.edf","data_Subject20_1_8.edf","data_Subject20_1_9.edf","data_Subject20_1_10.edf","data_Subject20_1_11.edf","data_Subject20_1_12.edf","data_Subject20_1_13.edf","data_Subject20_1_14.edf","data_Subject20_1_15.edf","data_Subject20_1_16.edf","data_Subject20_1_17.edf","data_Subject20_1_18.edf","data_Subject20_1_19.edf","data_Subject20_1_20.edf","data_Subject20_1_21.edf","data_Subject20_1_22.edf"]
filename22= ["data_Subject21_1_1.edf","data_Subject21_1_2.edf","data_Subject21_1_3.edf","data_Subject21_1_4.edf","data_Subject21_1_5.edf","data_Subject21_1_6.edf","data_Subject21_1_7.edf","data_Subject21_1_8.edf","data_Subject21_1_9.edf","data_Subject21_1_10.edf","data_Subject21_1_11.edf","data_Subject21_1_12.edf","data_Subject21_1_13.edf","data_Subject21_1_14.edf","data_Subject21_1_15.edf","data_Subject21_1_16.edf","data_Subject21_1_17.edf","data_Subject21_1_18.edf","data_Subject21_1_19.edf","data_Subject21_1_20.edf","data_Subject21_1_21.edf","data_Subject21_1_22.edf"]
filename23= ["data_Subject22_1_1.edf","data_Subject22_1_2.edf","data_Subject22_1_3.edf","data_Subject22_1_4.edf","data_Subject22_1_5.edf","data_Subject22_1_6.edf","data_Subject22_1_7.edf","data_Subject22_1_8.edf","data_Subject22_1_9.edf","data_Subject22_1_10.edf","data_Subject22_1_11.edf","data_Subject22_1_12.edf","data_Subject22_1_13.edf","data_Subject22_1_14.edf","data_Subject22_1_15.edf","data_Subject22_1_16.edf","data_Subject22_1_17.edf","data_Subject22_1_18.edf","data_Subject22_1_19.edf","data_Subject22_1_20.edf","data_Subject22_1_21.edf","data_Subject22_1_22.edf"]
filename24= ["data_Subject23_1_1.edf","data_Subject23_1_2.edf","data_Subject23_1_3.edf","data_Subject23_1_4.edf","data_Subject23_1_5.edf","data_Subject23_1_6.edf","data_Subject23_1_7.edf","data_Subject23_1_8.edf","data_Subject23_1_9.edf","data_Subject23_1_10.edf","data_Subject23_1_11.edf","data_Subject23_1_12.edf","data_Subject23_1_13.edf","data_Subject23_1_14.edf","data_Subject23_1_15.edf","data_Subject23_1_16.edf","data_Subject23_1_17.edf","data_Subject23_1_18.edf","data_Subject23_1_19.edf","data_Subject23_1_20.edf","data_Subject23_1_21.edf","data_Subject23_1_22.edf"]
filename25= ["data_Subject24_1_1.edf","data_Subject24_1_2.edf","data_Subject24_1_3.edf","data_Subject24_1_4.edf","data_Subject24_1_5.edf","data_Subject24_1_6.edf","data_Subject24_1_7.edf","data_Subject24_1_8.edf","data_Subject24_1_9.edf","data_Subject24_1_10.edf","data_Subject24_1_11.edf","data_Subject24_1_12.edf","data_Subject24_1_13.edf","data_Subject24_1_14.edf","data_Subject24_1_15.edf","data_Subject24_1_16.edf","data_Subject24_1_17.edf","data_Subject24_1_18.edf","data_Subject24_1_19.edf","data_Subject24_1_20.edf","data_Subject24_1_21.edf","data_Subject24_1_22.edf"]
filename26= ["data_Subject26_1_1.edf","data_Subject26_1_2.edf","data_Subject26_1_3.edf","data_Subject26_1_4.edf","data_Subject26_1_5.edf","data_Subject26_1_6.edf","data_Subject26_1_7.edf","data_Subject26_1_8.edf","data_Subject26_1_9.edf","data_Subject26_1_10.edf","data_Subject26_1_11.edf","data_Subject26_1_12.edf","data_Subject26_1_13.edf","data_Subject26_1_14.edf","data_Subject26_1_15.edf","data_Subject26_1_16.edf","data_Subject26_1_17.edf","data_Subject26_1_18.edf","data_Subject26_1_19.edf","data_Subject26_1_20.edf","data_Subject26_1_21.edf","data_Subject26_1_22.edf"]
filename27= ["data_Subject26_1_1.edf","data_Subject26_1_2.edf","data_Subject26_1_3.edf","data_Subject26_1_4.edf","data_Subject26_1_5.edf","data_Subject26_1_6.edf","data_Subject26_1_7.edf","data_Subject26_1_8.edf","data_Subject26_1_9.edf","data_Subject26_1_10.edf","data_Subject26_1_11.edf","data_Subject26_1_12.edf","data_Subject26_1_13.edf","data_Subject26_1_14.edf","data_Subject26_1_15.edf","data_Subject26_1_16.edf","data_Subject26_1_17.edf","data_Subject26_1_18.edf","data_Subject26_1_19.edf","data_Subject26_1_20.edf","data_Subject26_1_21.edf","data_Subject26_1_22.edf"]
filename28= ["data_Subject27_1_1.edf","data_Subject27_1_2.edf","data_Subject27_1_3.edf","data_Subject27_1_4.edf","data_Subject27_1_5.edf","data_Subject27_1_6.edf","data_Subject27_1_7.edf","data_Subject27_1_8.edf","data_Subject27_1_9.edf","data_Subject27_1_10.edf","data_Subject27_1_11.edf","data_Subject27_1_12.edf","data_Subject27_1_13.edf","data_Subject27_1_14.edf","data_Subject27_1_15.edf","data_Subject27_1_16.edf","data_Subject27_1_17.edf","data_Subject27_1_18.edf","data_Subject27_1_19.edf","data_Subject27_1_20.edf","data_Subject27_1_21.edf","data_Subject27_1_22.edf"]
filename29= ["data_Subject28_1_1.edf","data_Subject28_1_2.edf","data_Subject28_1_3.edf","data_Subject28_1_4.edf","data_Subject28_1_5.edf","data_Subject28_1_6.edf","data_Subject28_1_7.edf","data_Subject28_1_8.edf","data_Subject28_1_9.edf","data_Subject28_1_10.edf","data_Subject28_1_11.edf","data_Subject28_1_12.edf","data_Subject28_1_13.edf","data_Subject28_1_14.edf","data_Subject28_1_15.edf","data_Subject28_1_16.edf","data_Subject28_1_17.edf","data_Subject28_1_18.edf","data_Subject28_1_19.edf","data_Subject28_1_20.edf","data_Subject28_1_21.edf","data_Subject28_1_22.edf"]
filename30= ["data_Subject29_1_1.edf","data_Subject29_1_2.edf","data_Subject29_1_3.edf","data_Subject29_1_4.edf","data_Subject29_1_5.edf","data_Subject29_1_6.edf","data_Subject29_1_7.edf","data_Subject29_1_8.edf","data_Subject29_1_9.edf","data_Subject29_1_10.edf","data_Subject29_1_11.edf","data_Subject29_1_12.edf","data_Subject29_1_13.edf","data_Subject29_1_14.edf","data_Subject29_1_15.edf","data_Subject29_1_16.edf","data_Subject29_1_17.edf","data_Subject29_1_18.edf","data_Subject29_1_19.edf","data_Subject29_1_20.edf","data_Subject29_1_21.edf","data_Subject29_1_22.edf"]
filename31= ["data_Subject30_1_1.edf","data_Subject30_1_2.edf","data_Subject30_1_3.edf","data_Subject30_1_4.edf","data_Subject30_1_5.edf","data_Subject30_1_6.edf","data_Subject30_1_7.edf","data_Subject30_1_8.edf","data_Subject30_1_9.edf","data_Subject30_1_10.edf","data_Subject30_1_11.edf","data_Subject30_1_12.edf","data_Subject30_1_13.edf","data_Subject30_1_14.edf","data_Subject30_1_15.edf","data_Subject30_1_16.edf","data_Subject30_1_17.edf","data_Subject30_1_18.edf","data_Subject30_1_19.edf","data_Subject30_1_20.edf","data_Subject30_1_21.edf","data_Subject30_1_22.edf"]
filename32= ["data_Subject31_1_1.edf","data_Subject31_1_2.edf","data_Subject31_1_3.edf","data_Subject31_1_4.edf","data_Subject31_1_5.edf","data_Subject31_1_6.edf","data_Subject31_1_7.edf","data_Subject31_1_8.edf","data_Subject31_1_9.edf","data_Subject31_1_10.edf","data_Subject31_1_11.edf","data_Subject31_1_12.edf","data_Subject31_1_13.edf","data_Subject31_1_14.edf","data_Subject31_1_15.edf","data_Subject31_1_16.edf","data_Subject31_1_17.edf","data_Subject31_1_18.edf","data_Subject31_1_19.edf","data_Subject31_1_20.edf","data_Subject31_1_21.edf","data_Subject31_1_22.edf"]
filename33= ["data_Subject32_1_1.edf","data_Subject32_1_2.edf","data_Subject32_1_3.edf","data_Subject32_1_4.edf","data_Subject32_1_5.edf","data_Subject32_1_6.edf","data_Subject32_1_7.edf","data_Subject32_1_8.edf","data_Subject32_1_9.edf","data_Subject32_1_10.edf","data_Subject32_1_11.edf","data_Subject32_1_12.edf","data_Subject32_1_13.edf","data_Subject32_1_14.edf","data_Subject32_1_15.edf","data_Subject32_1_16.edf","data_Subject32_1_17.edf","data_Subject32_1_18.edf","data_Subject32_1_19.edf","data_Subject32_1_20.edf","data_Subject32_1_21.edf","data_Subject32_1_22.edf"]
filename34= ["data_Subject33_1_1.edf","data_Subject33_1_2.edf","data_Subject33_1_3.edf","data_Subject33_1_4.edf","data_Subject33_1_5.edf","data_Subject33_1_6.edf","data_Subject33_1_7.edf","data_Subject33_1_8.edf","data_Subject33_1_9.edf","data_Subject33_1_10.edf","data_Subject33_1_11.edf","data_Subject33_1_12.edf","data_Subject33_1_13.edf","data_Subject33_1_14.edf","data_Subject33_1_15.edf","data_Subject33_1_16.edf","data_Subject33_1_17.edf","data_Subject33_1_18.edf","data_Subject33_1_19.edf","data_Subject33_1_20.edf","data_Subject33_1_21.edf","data_Subject33_1_22.edf"]
filename35= ["data_Subject34_1_1.edf","data_Subject34_1_2.edf","data_Subject34_1_3.edf","data_Subject34_1_4.edf","data_Subject34_1_5.edf","data_Subject34_1_6.edf","data_Subject34_1_7.edf","data_Subject34_1_8.edf","data_Subject34_1_9.edf","data_Subject34_1_10.edf","data_Subject34_1_11.edf","data_Subject34_1_12.edf","data_Subject34_1_13.edf","data_Subject34_1_14.edf","data_Subject34_1_15.edf","data_Subject34_1_16.edf","data_Subject34_1_17.edf","data_Subject34_1_18.edf","data_Subject34_1_19.edf","data_Subject34_1_20.edf","data_Subject34_1_21.edf","data_Subject34_1_22.edf"]
filename36= ["data_Subject35_1_1.edf","data_Subject35_1_2.edf","data_Subject35_1_3.edf","data_Subject35_1_4.edf","data_Subject35_1_5.edf","data_Subject35_1_6.edf","data_Subject35_1_7.edf","data_Subject35_1_8.edf","data_Subject35_1_9.edf","data_Subject35_1_10.edf","data_Subject35_1_11.edf","data_Subject35_1_12.edf","data_Subject35_1_13.edf","data_Subject35_1_14.edf","data_Subject35_1_15.edf","data_Subject35_1_16.edf","data_Subject35_1_17.edf","data_Subject35_1_18.edf","data_Subject35_1_19.edf","data_Subject35_1_20.edf","data_Subject35_1_21.edf","data_Subject35_1_22.edf"]

filename.append(filename1)
filename.append(filename2)
filename.append(filename3)
filename.append(filename4)
filename.append(filename5)
filename.append(filename6)
filename.append(filename7)
filename.append(filename8)
filename.append(filename9)
filename.append(filename10)
filename.append(filename11)
filename.append(filename12)
filename.append(filename13)
filename.append(filename14)
filename.append(filename15)
filename.append(filename16)
filename.append(filename17)
filename.append(filename18)
filename.append(filename19)
filename.append(filename20)
filename.append(filename21)
filename.append(filename22)
filename.append(filename23)
filename.append(filename24)
filename.append(filename25)
filename.append(filename26)
filename.append(filename27)
filename.append(filename28)
filename.append(filename29)
filename.append(filename30)
filename.append(filename31)
filename.append(filename32)
filename.append(filename33)
filename.append(filename34)
filename.append(filename35)
filename.append(filename36)

fn=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36']


import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances   
sns.set(style='white', rc={'figure.figsize':(12,8)})
np.random.seed(42)

tot_maps=[]   
spikes=[]
lin_reg=[]



pcorr = np.array([])
gfpumap =  np.array([])
gfpold =  np.array([])  # microstate label sequence for each k-means run
CE_array = np.array([])
totsum1=np.array([])
totsum2=np.array([])
totsum3=np.array([])
totsum4=np.array([])
totsum5=np.array([])
totsum6=np.array([])
totsum7=np.array([])
totsum8=np.array([])
totsum9=np.array([])
totsum10=np.array([])
 






mapspca1=np.array([])
L_ind=np.array([])
arr =np.array([])  
datasto =np.array([])
L_individual =np.array([],dtype='i')  
L_group=np.array([],dtype='i') 
sum_rec=[]




# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:23:43 2021

@author: win
"""
# importing matplot lib
time1=[]
for t in range(500):
    for i in range(19):
     time1.append([t,i])



time2=[]
for t in range(0,2000,1):
    for i in range(19):
     time2.append([t*0.25,i])
    
elec=[]
for t in range(19):
    elec.append(t)
time1=np.resize(time1,(9500,2))    
elec=np.array(elec)     
time2=np.resize(time2,(38000,2))     


from scipy.io import savemat

mdic = {"t1": time1}
savemat("time1.mat", mdic)

mdic = {"t2": time2}
savemat("time2.mat", mdic)


# from scipy.interpolate import interp1d

# channels=['Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P7','P8','P3','P4','O1','O2','Fz','Cz','Pz']
# locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-4.7,  6.2,  8. ],[ 4.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-4.7, -6.2,  8. ],[ 4.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
# nmaps=4
# filenamealz=[]
# filenamealz1 = ['AD1.edf','AD2.edf','AD3.edf','AD4.edf','AD5.edf','AD6.edf','AD7.edf','AD8.edf']
# #filenamealz2 = ["data_Subject00_1_1.edf","data_Subject01_1_1.edf","data_Subject02_1_1.edf","data_Subject03_1_1.edf","data_Subject04_1_1.edf","data_Subject05_1_1.edf","data_Subject06_1_1.edf","data_Subject07_1_1.edf","data_Subject08_1_1.edf","data_Subject09_1_1.edf","data_Subject10_1_1.edf","data_Subject11_1_1.edf","data_Subject12_1_1.edf","data_Subject13_1_1.edf","data_Subject14_1_1.edf","data_Subject15_1_1.edf","data_Subject16_1_1.edf","data_Subject17_1_1.edf","data_Subject18_1_1.edf","data_Subject19_1_1.edf","data_Subject20_1_1.edf","data_Subject21_1_1.edf","data_Subject22_1_1.edf","data_Subject23_1_1.edf","data_Subject24_1_1.edf","data_Subject25_1_1.edf","data_Subject26_1_1.edf","data_Subject27_1_1.edf","data_Subject28_1_1.edf","data_Subject29_1_1.edf","data_Subject30_1_1.edf"]
# filenamealz.append(filenamealz1)

# mbappe=[]
# arr =np.array([])  
# L_individual =np.array([],dtype='i')  
# L_group=np.array([],dtype='i')  
# for i in range(1):
    
#     for j in range(7):
#         chs, fs, data_raw,ch,n_per_rec = read_edf(filenamealz[i][j])
#         data_raw=data_raw[:,0:19]
#         #for k in range()
#         chs=chs[0:19]
#         data = bp_filter(data_raw, f_lo=8, f_hi=16, fs=fs)
#         data=data[0:2560]
#         data=data.T
#         mbappe1=[]
#         for num in range(19):
            
#             data1=data[num]
#             x = np.linspace(0,9999, num=2560, endpoint=True)
#             y = (data1).T
#             f2 = interp1d(x, y, kind='cubic')
#             xnew = np.linspace(0,9999, num=10000, endpoint=True)
#             datan=f2(xnew)
#             mbappe1.append(datan)
#         mbappe1=np.array(mbappe1)
#         mbappe1=np.resize(mbappe1,(19,10000))
#         mbappe.append(mbappe1.T)
# mbappe=np.resize(mbappe,(7,10000,19))
 





from scipy.io import loadmat
annots = loadmat('alzdata.mat')
con_list = [[element for element in upperElement] for upperElement in annots['alzdata']]
con=np.array(con_list)
mbappe=con 
 
score=[] 
 
 
for i1 in range(31): 
    def locmax(x):
        """Get local maxima of 1D-array

        Args:
            x: numeric sequence
        Returns:
            m: list, 1D-indices of local maxima
        """

        dx = np.diff(x) # discrete 1st derivative
        zc = np.diff(np.sign(dx)) # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0] # indices of local max.
        return m
    
    for j1 in range(2):
    
            chs, fs, data_raw = read_edf(filename[i1][j1])
            data = bp_filter(data_raw, f_lo=1, f_hi=35, fs=fs)
            datasto=np.append(datasto,data)
    score.append(1)       
            
for i1 in range(7): 
                
       for nm in range(4): 
            mb=mbappe[:,0+nm*2000:2000+nm*2000,:]  
            datasto=np.append(datasto,mb[i1])            
            score.append(0) 




for i1 in range(31,36): 
    def locmax(x):
        """Get local maxima of 1D-array

        Args:
            x: numeric sequence
        Returns:
            m: list, 1D-indices of local maxima
        """

        dx = np.diff(x) # discrete 1st derivative
        zc = np.diff(np.sign(dx)) # zero-crossings of dx
        m = 1 + np.where(zc == -2)[0] # indices of local max.
        return m
    
    for j1 in range(2):
    
            chs, fs, data_raw = read_edf(filename[i1][j1])
            data = bp_filter(data_raw, f_lo=1, f_hi=35, fs=fs)
            datasto=np.append(datasto,data)

    score.append(1) 
    
    
    
    
    
datasto=np.resize(datasto,(128000,19))   
data=datasto


score=np.array(score)
            
for i in range(1):
                data_norm = data - data.mean(axis=1, keepdims=True)
                data_norm /= data_norm.std(axis=1, keepdims=True)
                # --- GFP peaks ---
                gfp = np.nanstd(data, axis=1)
                gfp2 = np.sum(gfp**2) # normalizing constant
                gfp_peaks = locmax(gfp)
                
                gfp_values = gfp[gfp_peaks]
                gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
                n_gfp = gfp_peaks.shape[0]
                V = data[gfp_peaks, :]
                sumV2 = np.sum(V**2)
                
                
                n_runs=10
                maxerr=1e-6
                maxiter=500
                
                mode = ["aahc", "kmeans"][1]
                print(f"Clustering algorithm: {mode:s}")
                n_maps = 4
                
                locs = []        
                
                n_grid=68
                locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-8.7,  6.2,  8. ],[ 8.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-8.7, -6.2,  8. ],[ 8.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
                
                interpol=False
                doplot=True
                n_win=3
                
                # maps1, x1, gfp_peaks1, gev1 = clustering(data, fs, chs, locs, mode, n_maps, interpol=False, doplot=True)
                
                regreco1=[]
                regreco2=[]
                regreco3=[]
                regreco4=[]
                regreco5=[]
                regreco6=[]
                regreco7=[]
                regreco8=[]
                regreco9=[]
                regreco10=[]
              
        
        
            
                sum1=[]
                sum4=[]
                sum3=[]
                sum2=[]
                sum5=[]
                sum6=[]
                sum7=[]
                sum8=[]
                sum9=[]
                sum10=[]
                
Cpred_dtc=[]
Cscore_dtc=[]

Cpred_rf=[]
Cscore_rf=[]

Cpred_svm=[]
Cscore_svm=[]






Ccorpred_dtc=[]
Ccorscore_dtc=[]

Ccorpred_rf=[]
Ccorscore_rf=[]

Ccorpred_svm=[]
Ccorscore_svm=[]






 
covarregrecor1=np.array([])
covarregrecor2=np.array([])
                            
covarregrecor5=np.array([])
covarregrecor6=np.array([])
                            
covarregrecor9=np.array([])
covarregrecor10=np.array([])
                            
regrecoeffstack=[]                          
                            
                            
                            
                            
for nmaps in range(18,101):
                            regreco1=[]
                            regreco2=[]
                            regreco3=[]
                            regreco4=[]
                            regreco5=[]
                            regreco6=[]
                            regreco7=[]
                            regreco8=[]
                            regreco9=[]
                            regreco10=[]
                            import numpy as np
                            from sklearn.datasets import make_sparse_coded_signal
                            from sklearn.decomposition import DictionaryLearning

                            dict_learner = DictionaryLearning(n_components=nmaps, transform_algorithm='lasso_lars', random_state=42)
                            X1 = dict_learner.fit_transform(data_norm.T)            
                            regression_score=[]
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            from sklearn import linear_model
                            reg = linear_model.LassoLars(alpha=0.01, normalize=False)
                            for j in range(data_norm.shape[0]):
                                y1 = data_norm[j]
                                reg = LinearRegression().fit(X1, y1)                                
                                regreco1.append(reg.coef_)
                            regrecoeffstack.append(X1)
                
               
                            def locmax(x):
                                """Get local maxima of 1D-array

                                Args:
                                    x: numeric sequence
                                Returns:
                                    m: list, 1D-indices of local maxima
                                """

                                dx = np.diff(x) # discrete 1st derivative
                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                return m
                            # maps1, x1, gfp_peaks1, gev1 = clustering(data, fs, chs, locs, "pca",nmaps, interpol=False, doplot=True)
                            # X1 = maps1.T        
                            # regression_score=[]
                            # import numpy as np
                            # from sklearn.linear_model import LinearRegression
                            # #regression_score=[] 
                            # # X1 = m2.T
                            # min_maps=22
                            # for j in range(data_norm.shape[0]):
                            #     y1 = data_norm[j]
                            #     reg = LinearRegression().fit(X1, y1)                                
                            #     regreco4.append(reg.coef_)
                
              
                            def locmax(x):
                                """Get local maxima of 1D-array

                                Args:
                                    x: numeric sequence
                                Returns:
                                    m: list, 1D-indices of local maxima
                                """

                                dx = np.diff(x) # discrete 1st derivative
                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                return m
                            # maps1, x1, gfp_peaks1, gev1 = clustering(data, fs, chs, locs, "ica", nmaps, interpol=False, doplot=True)
                            # X1 = maps1.T
                            # regression_score=[]
                            # import numpy as np
                            # from sklearn.linear_model import LinearRegression
                            # #regression_score=[] 
                            # # X1 = m2.T
                            # min_maps=22
                            # for j in range(data_norm.shape[0]):
                            #     y1 = data_norm[j]
                            #     reg = LinearRegression().fit(X1, y1)                                
                            #     regreco3.append(reg.coef_)
                                
                                
                                
                            def locmax(x):
                                """Get local maxima of 1D-array

                                Args:
                                    x: numeric sequence
                                Returns:
                                    m: list, 1D-indices of local maxima
                                """

                                dx = np.diff(x) # discrete 1st derivative
                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                return m
                            maps1, x1, gfp_peaks1, gev1 = clustering(data, fs, chs, locs, "kmeans",nmaps, interpol=False, doplot=True)
                            X1 = maps1.T
                            regrecoeffstack.append(X1)
                            regression_score=[]
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            #regression_score=[] 
                            # X1 = m2.T
                            for j in range(data_norm.shape[0]):
                                y1 = data_norm[j]
                                reg.fit(X1, y1)                                
                                regreco2.append(reg.coef_)      
            
            
            
            
            
            
            
            
            
            
             
            
            
        
        
        
        
        
        
        
        
                    
                            def locmax(x):
                                """Get local maxima of 1D-array
            
                                Args:
                                    x: numeric sequence
                                Returns:
                                    m: list, 1D-indices of local maxima
                                """
            
                                dx = np.diff(x) # discrete 1st derivative
                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                return m
                            data_norm = data - data.mean(axis=1, keepdims=True)
                            data_norm /= data_norm.std(axis=1, keepdims=True)
                            # --- GFP peaks ---
                            gfp = np.nanstd(data, axis=1)
                            gfp2 = np.sum(gfp**2) # normalizing constant
                            gfp_peaks = locmax(gfp)
                            
                            gfp_values = gfp[gfp_peaks]
                            gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
                            n_gfp = gfp_peaks.shape[0]
                            V = data[gfp_peaks, :]
                            sumV2 = np.sum(V**2)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            def locmax(x):
                                            """Get local maxima of 1D-array
            
                                            Args:
                                                x: numeric sequence
                                            Returns:
                                                m: list, 1D-indices of local maxima
                                            """
            
                                            dx = np.diff(x) # discrete 1st derivative
                                            zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                            m = 1 + np.where(zc == -2)[0] # indices of local max.
                                            return m
                            n_runs=10
                            maxerr=1e-6
                            maxiter=500
                            
                            mode = ["aahc", "kmeans"][1]
                            print(f"Clustering algorithm: {mode:s}")
                            n_maps = 4
                            
                            locs = []        
                            
                            n_grid=68
                            locs=[[-2.7,  8.6,  3.6],[ 2.7,  8.6,  3.6],[-8.7,  6.2,  8. ],[ 8.7,  6.2,  8. ],[-6.7,  5.2,  3.6],[ 6.7,  5.2,  3.6],[-7.8,  0. ,  3.6],[ 7.8,  0. ,  3.6],[-6.1,  0. ,  9.7],[ 6.1,  0. ,  9.7],[-7.3, -2.5,  0. ],[ 7.3, -2.5,  0. ],[-8.7, -6.2,  8. ],[ 8.7, -6.2,  8. ],[-2.7, -8.6,  3.6],[ 2.7, -8.6,  3.6],[ 0. ,  6.7,  9.5],[ 0. ,  0. , 12. ],[ 0. , -6.7,  9.5]]
                            
                            interpol=False
                            doplot=True
                            n_win=3
                            
                            # maps1, x1, gfp_peaks1, gev1 = clustering(data, fs, chs, locs, mode, n_maps, interpol=False, doplot=True)
                            
                            
            
                            
                            mi=[]
                            mini=[]
                            spikes=[]
                            lows=[]
                            locmax=[]
                            locmin=[]
                            
                            
                            for i in range(V.shape[0]):
                                x1=V[i]
                                x=np.zeros((5,5))
                                x[0][0]=x1[0]
                                x[0][1]=x1[0]
                                x[0][2]=(x1[0]+x1[1])/2
                                x[0][3]=x1[1]
                                x[0][4]=x1[1]
                                x[1][0]=x1[4]
                                x[1][1]=x1[2]
                                x[1][2]=x1[16]
                                x[1][3]=x1[3]
                                x[1][4]=x1[5]
                                x[2][0]=x1[6]
                                x[2][1]=x1[8]
                                x[2][2]=x1[17]
                                x[2][3]=x1[9]
                                x[2][4]=x1[7]
                                x[3][0]=x1[10]
                                x[3][1]=x1[12]
                                x[3][2]=x1[18]
                                x[3][3]=x1[13]
                                x[3][4]=x1[11]
                                x[4][0]=x1[14]
                                x[4][1]=x1[14]
                                x[4][2]=(x1[14]+x1[15])/2
                                x[4][3]=x1[15]
                                x[4][4]=x1[15]
                                
                                test=1
                                r=x.shape[0]
                                c=x.shape[1]
                                
                                m=np.zeros((r,c))
                                maxi=float(-10000)
                                mini=float(10000)
                                t=0
                                lo=0
                                for i in range(r):
                                    for j in range(c):
                                        maxi=float(-10000)
                                        mini=float(10000)
                                        maxip=[]    
                                        for i1 in range(i-1,i+2):
                                            for j1 in range(j-1,j+2):
                                                if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                                                    a=0
                                                    test=8
                                                else:
                                                    test=0
                                                    if(maxi<float(x[i1][j1])):
                                                       maxi=float(x[i1][j1])
                                                    if(mini>float(x[i1][j1])):
                                                       mini=float(x[i1][j1])
                                        # locmax.append(maxi)
                                        # locmin.append(mini)
                                        # maxi=np.max(x[i-1][j-1],x[i][j-1],x[i+1][j-1],x[i-1][j],x[i+1][j],x[i-1][j+1],x[i][j],x[i+1][j-1])
                                        if(x[i][j]==maxi):
                                            m[i][j]=1
                                            t=t+1
                                        if(x[i][j]==mini):
                                            m[i][j]=-1
                                            lo=lo+1
                                x[0][2]=x1[1]
                                i=1
                                j=3
                                for i1 in range(i-1,i+2):
                                    for j1 in range(j-1,j+2):
                                        if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                                            a=0
                                            test=8
                                        else:
                                            test=0
                                            if(maxi<float(x[i1][j1])):
                                               maxi=float(x[i1][j1])
                                            if(mini>float(x[i1][j1])):
                                               mini=float(x[i1][j1])
                                if(x[i][j]==maxi):
                                    m[i][j]=1
                                if(x[i][j]==mini):
                                    m[i][j]=-1
                                    
                                
                                    
                                
                                
                                x[0][2]=x1[0]
                                i=1
                                j=1
                                for i1 in range(i-1,i+2):
                                    for j1 in range(j-1,j+2):
                                        if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                                            a=0
                                            test=8
                                        else:
                                            test=0
                                            if(maxi<float(x[i1][j1])):
                                               maxi=float(x[i1][j1])
                                            if(mini>float(x[i1][j1])):
                                               mini=float(x[i1][j1])
                                if(x[i][j]==maxi):
                                    m[i][j]=1
                                if(x[i][j]==mini):
                                    m[i][j]=-1
                                    
                                    
                                    
                                    
                                x[4][2]=x1[15]
                                i=3
                                j=3
                                for i1 in range(i-1,i+2):
                                    for j1 in range(j-1,j+2):
                                        if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                                            a=0
                                            test=8
                                        else:
                                            test=0
                                            if(maxi<float(x[i1][j1])):
                                               maxi=float(x[i1][j1])
                                            if(mini>float(x[i1][j1])):
                                               mini=float(x[i1][j1])
                                if(x[i][j]==maxi):
                                    m[i][j]=1
                                if(x[i][j]==mini):
                                    m[i][j]=-1
                                    
                                
                                    
                                
                                
                                x[4][2]=x1[14]
                                i=3
                                j=1
                                for i1 in range(i-1,i+2):
                                    for j1 in range(j-1,j+2):
                                        if(i1<0) or (i1>r-1) or (j1<0) or (j1>c-1):
                                            a=0
                                            test=8
                                        else:
                                            test=0
                                            if(maxi<float(x[i1][j1])):
                                               maxi=float(x[i1][j1])
                                            if(mini>float(x[i1][j1])):
                                               mini=float(x[i1][j1])
                                if(x[i][j]==maxi):
                                    m[i][j]=1
                                if(x[i][j]==mini):
                                    m[i][j]=-1
                                    
                                    
                                    
                                    
                                    
                                    
                                mi.append(m)
                                spikes.append(t)
                                lows.append(lo)
                            
                            m1=np.resize(mi,(V.shape[0],25))
                            d=[1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,23]
                            m2=(m1.T)[d]
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            s=m2.T
                            single_sourcep=[]
                            single_sourcen=[]
                            
                            for i in range(V.shape[0]):
                                p=0
                                n=0
                                for j in range(19):
                                    if(m2[j][i]==1):
                                        p=p+1
                                    if(m2[j][i]==-1):
                                        n=n+1
                            
                                if(p<2):
                                    if(p>0):
                                        if(n==1):
                                            upi=1
                                        else:
                                            single_sourcep.append(i)
                                if(n<2):
                                    if(n>0):
                                        if(p==1):
                                            upi=1
                                        else:
                                            single_sourcep.append(i)
                            
                            from sklearn.metrics.pairwise import cosine_similarity,cosine_distances   
                            
                            n_maps=7
                            n_runs=500
                            n_ch=19
                            CVtotal1=[]
                            mapsset1=[]
                            if(1):
                                
                                V = V[single_sourcep]
                                n_gfp=V.shape[0]
                                n_t=V.shape[0]
                                sumV2 = np.sum(V**2)
                            sum5=[]
                            n_maps=nmaps
                                # store results for each k-means run
                            cv_list =   []  # cross-validation criterion for each k-means run
                            gev_list =  []  # GEV of each map for each k-means run
                            gevT_list = []  # total GEV values for each k-means run
                            maps_list = []  # microstate maps for each k-means run
                            L_list =    []  # microstate label sequence for each k-means run
                            for run in range(n_runs):
                                    # initialize random cluster centroids (indices w.r.t. n_gfp)
                                    rndi = np.random.permutation(n_gfp)[:n_maps]
                                    maps = V[rndi, :]
                                    # normalize row-wise (across EEG channels)
                                    maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
                                    # initialize
                                    n_iter = 0
                                    var0 = 1.0
                                    var1 = 0.0
                                    # convergence criterion: variance estimate (step 6)
                                    while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
                                        
                                        from sklearn.metrics.pairwise import cosine_similarity,cosine_distances   
                                        C=abs(cosine_similarity(V, maps))
                                        L = np.argmax(C**2, axis=1)
                                        # (step 8)
                                        for k in range(n_maps):
                                            Vt = V[L==k, :]
                                            # (step 8a)
                                            Sk = np.dot(Vt.T, Vt)
                                            # (step 8b)
                                            evals, evecs = np.linalg.eig(Sk)
                                            v = evecs[:, np.argmax(np.abs(evals))]
                                            v = v.real
                                            maps[k, :] = v/np.sqrt(np.sum(v**2))
                                        # (step 5)
                                        var1 = var0
                                        var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
                                        var0 /= (n_gfp*(n_ch-1))
                                        n_iter += 1
                                    if (n_iter < maxiter):
                                        print((f"\tK-means run {run+1:d}/{n_runs:d} converged after "
                                               f"{n_iter:d} iterations."))
                                    else:
                                        print((f"\tK-means run {run+1:d}/{n_runs:d} did NOT converge "
                                               f"after {maxiter:d} iterations."))
                            
                                    # CROSS-VALIDATION criterion for this run (step 8)
                                    
                                    C_=abs(cosine_similarity(V, maps))
                                    L_ = np.argmax(C_**2, axis=1)
                                    var = np.sum(V**2) - np.sum(np.sum(maps[L_, :]*V, axis=1)**2)
                                    var /= (n_t*(n_ch-1))
                                    cv = var
                            
                                    # GEV (global explained variance) of cluster k
                                    gev = np.zeros(n_maps)
                                    for k in range(n_maps):
                                        r = L==k
                                        gev[k] = 0
                                    gev_total = np.sum(gev)
                            
                                    # store
                                    cv_list.append(cv)
                                    gev_list.append(gev)
                                    gevT_list.append(gev_total)
                                    maps_list.append(maps)
                                    L_list.append(L_)
                            
                            # select best run
                            k_opt = np.argmin(cv_list)
                            #k_opt = np.argmax(gevT_list)
                            maps = maps_list[k_opt]
                            # ms_gfp = ms_list[k_opt] # microstate sequence at GFP peaks
                            gev = gev_list[k_opt]
                            L_ = L_list[k_opt]
                            CVtotal1.append(np.min(cv_list))
                            mapsset1.append(maps)
                        
                            plt.plot(CVtotal1)
                            plt.show()
                            
                            regression_score=[]
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            X1=maps.T
                            # cm = plt.cm.seismic
                            # fig, axarr = plt.subplots(1, i, figsize=(20,5))
                            # for imap in range(n_maps):
                            #     axarr[imap].imshow(eeg2map((X1.T)[imap, :]), cmap=cm, origin='lower')
                            #     axarr[imap].set_xticks([])
                            #     axarr[imap].set_xticklabels([])
                                #     axarr[imap].set_yticks([])
                            #     axarr[imap].set_yticklabels([])
                            # plt.plot()
                            #regression_score=[] 
                            # X1 = m2.T
                            regrecoeffstack.append(X1)
                            for j in range(data_norm.shape[0]):
                                y1 = data_norm[j]
                                reg = LinearRegression().fit(X1, y1)                                
                                regreco5.append(reg.coef_)
                            
                           
                            for i in range(1):
                                            import numpy as np
                                            from sklearn.datasets import make_sparse_coded_signal
                                            from sklearn.decomposition import DictionaryLearning
            
                                            dict_learner = DictionaryLearning(n_components=nmaps, transform_algorithm='lasso_lars', random_state=42)
                                            X1 = dict_learner.fit_transform(V.T) 
                                            regrecoeffstack.append(X1)
                                            regrecoeffstack.append(X1)
                                            regression_score=[]
                                            import numpy as np
                                            from sklearn.linear_model import LinearRegression
                                            
                                     
                                            
                                            for j in range(data_norm.shape[0]):
                                                y1 = data_norm[j]
                                                reg = LinearRegression().fit(X1, y1)                                
                                                regreco6.append(reg.coef_)
                              
                            for i in range(1):
                                            def locmax(x):
                                            
                                                dx = np.diff(x) # discrete 1st derivative
                                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                                return m
                                            # maps1, x1, gfp_peaks1, gev1 = clustering(V, fs, chs, locs, "pca", nmaps, interpol=False, doplot=True)
                                            # X1 = maps1.T        
                                            # regression_score=[]
                                            # import numpy as np
                                            # from sklearn.linear_model import LinearRegression
                                            # #regression_score=[] 
                                            # # X1 = m2.T
                                            # min_maps=22
                                            # for j in range(V.shape[0]):
                                            #     y1 = data_norm[j]
                                            #     reg = LinearRegression().fit(X1, y1)                                
                                            #     regreco7.append(reg.coef_)
                               
                            for i in range(1):
                                            def locmax(x):
                                                """Get local maxima of 1D-array
                
                                                Args:
                                                    x: numeric sequence
                                                Returns:
                                                    m: list, 1D-indices of local maxima
                                                """
                                                dx = np.diff(x) # discrete 1st derivative
                                                zc = np.diff(np.sign(dx)) # zero-crossings of dx
                                                m = 1 + np.where(zc == -2)[0] # indices of local max.
                                                return m
                                            # maps1, x1, gfp_peaks1, gev1 = clustering(V, fs, chs, locs, "ica", nmaps, interpol=False, doplot=True)
                                            # X1 = maps1.T
                                            # regression_score=[]
                                            # import numpy as np
                                            # from sklearn.linear_model import LinearRegression
                                            # #regression_score=[] 
                                            # # X1 = m2.T
                                            # min_maps=22
                                            # for j in range(V.shape[0]):
                                            #     y1 = data_norm[j]
                                            #     reg = LinearRegression().fit(X1, y1)                                
                                            #     regreco8.append(reg.coef_)
                                
                                
                              
                            for i in range(1):
                                            
                                            import numpy as np
                                            import scipy as sp
                                            from sklearn.linear_model import OrthogonalMatchingPursuit
            
                                            class KSVD:
                                                def __init__(self,rank,num_of_NZ=None,func_svd=False,
                                                             max_iter = 20,max_tol = 1e-12):
                                                    self.rank = rank
                                                    self.max_iter = max_iter
                                                    self.max_tol = max_tol
                                                    self.num_of_NZ = num_of_NZ
                                                    self.func_svd = func_svd
            
                                                def _initialize_parameters(self,Y):
                                                    A = np.random.randn(Y.shape[0],self.rank)
                                                    X = np.zeros((self.rank,Y.shape[1]))
            
                                                    return A, X
            
                                                def _estimate_X(self,Y,A):
                                                    if self.num_of_NZ is None:
                                                        n_nonzero_coefs = np.ceil(0.1 * A.shape[1])
                                                    else:
                                                        n_nonzero_coefs = self.num_of_NZ
            
                                                    omp = OrthogonalMatchingPursuit(n_nonzero_coefs = int(n_nonzero_coefs))
                                                    for j in range(A.shape[1]):
                                                        A[:,j] /= max(np.linalg.norm(A[:,j]),1e-20)
                                                        
                                                    omp.fit(A,Y)
                                                    return omp.coef_.T
            
                                                def _update_parameters(self,Y,A,X):
                                                    for j in range(self.rank):
                                                        NZ = np.where(X[j, :] != 0)[0]
                                                        A_tmp = A
                                                        X_tmp = X
                                                        if len(NZ) > 0:
                                                            A_tmp[:,j][:] = 0
                                                            E_R = Y[:,NZ]-A_tmp.dot(X_tmp[:,NZ])
            
                                                            if self.func_svd is True:
                                                                u, s, v = np.linalg.svd(E_R)
                                                                X[j,NZ] = s[0]*np.asarray(v)[0]
                                                                A[:,j] = u[:,0]
                                                            else:
                                                                A_R = E_R.dot(X[j,NZ].T)
                                                                A_R /= np.linalg.norm(A_R)
                                                                X_R = E_R.T.dot(A_R)
                                                                X[j,NZ] = X_R.T
                                                                A[:,j] = A_R
            
                                                    return A_tmp,X_tmp
            
                                                def _edit_dictionary(self,Y,A,X):
            
                                                    E = Y-A.dot(X)
                                                    E_atom_norm = np.linalg.norm(E,axis=0).tolist()
                                                    Max_index = E_atom_norm.index(max(E_atom_norm))
                                                    examp = np.matrix(Y[:,Max_index])
            
                                                    for j in range(A.shape[1]):
                                                        for k in range(j+1,A.shape[1]):
                                                            if np.linalg.norm(A[:,j]-A[:,k]) < 1e-1:
                                                                A[:,k]=examp+np.random.randn(1,Y.shape[0])*0.0001
            
                                                    for j in range(X.shape[0]):
                                                        if np.linalg.norm(X[j,:])/X.shape[1] < 1e-5:
                                                            A[:,j]=examp+np.random.randn(1,Y.shape[0])*0.0001
            
                                                    return A
            
                                                def fit(self, Y):
                                                    """
                                                    Y = AX
                                                    Y: shape = [n_features,n_samples]
                                                    A: Dictionary = [n_features, rank]
                                                    X: Sparse = [rank, n_samples]
                                                    """
                                                    err = 1e+8
                                                    err_f = 1e+10
                                                    A, X = self._initialize_parameters(Y)
                                                    for j in range(self.max_iter):
            
                                                        X_tmp = self._estimate_X(Y,A)
                                                        err = np.linalg.norm(Y-A.dot(X_tmp))
            
                                                        if err < self.max_tol:
                                                            break
                                                        if err < err_f :
                                                            err_f = err
                                                            X = X_tmp
                                                            print(j,':error=',err/(Y.shape[0]*Y.shape[1]))
                                                        A,X = self._update_parameters(Y,A,X)
                                                        A = self._edit_dictionary(Y,A,X)
            
                                                    return A, X    
                                                
                                            
            
                                            Y=data_norm
                                            ksvd = KSVD(rank=nmaps)
                                            A, X= ksvd.fit(Y)
                                            X1 = X.T
                                            regrecoeffstack.append(X1)
                                            regression_score=[]
                                            import numpy as np
                                            from sklearn.linear_model import LinearRegression
                                            #regression_score=[] 
                                            # X1 = m2.T
                                            min_maps=22                             
                                            for j in range(data_norm.shape[0]):
                                               y1 = data_norm[j]
                                               y1=np.resize(y1,(y1.shape[0],1))
                                               sim =cosine_similarity(X1.T,y1.T)                                  
                                               regreco9.append(sim)
                                
                                
                            s2=V.shape[0] 
                            for i in range(1):
                                            
                                            Y=V
                                            ksvd = KSVD(rank=nmaps)
                                            A, X= ksvd.fit(Y)
                                            X1 = X.T
                                            regrecoeffstack.append(X1)
                                            regression_score=[]
                                            import numpy as np
                                            from sklearn.linear_model import LinearRegression
                                            #regression_score=[] 
                                            # X1 = m2.T
                                            for j in range(data_norm.shape[0]):
                                                y1 = data_norm[j]
                                                reg = LinearRegression().fit(X1, y1)                                
                                                regreco10.append(reg.coef_)

                            regrecor1=np.resize(regreco1,(64,2000,nmaps))
                            regrecor2=np.resize(regreco2,(64,2000,nmaps))
                            
                            regrecor5=np.resize(regreco5,(64,2000,nmaps))
                            regrecor6=np.resize(regreco6,(64,2000,nmaps))
                            
                            regrecor9=np.resize(regreco9,(64,2000,nmaps))
                            regrecor10=np.resize(regreco10,(64,2000,nmaps))  
                            
                            
                            
                            
                            
                            regrecor1=np.array(regrecor1)
                            regrecor2=np.array(regrecor2)
                            
                            regrecor5=np.array(regrecor5)
                            regrecor6=np.array(regrecor6)
                            
                            regrecor9=np.array(regrecor9)
                            regrecor10=np.array(regrecor10)
                            
                            
                            regreco1=np.array(regreco1)
                            regreco2=np.array(regreco2)
                            
                            regreco5=np.array(regreco5)
                            regreco6=np.array(regreco6)
                            
                            regreco9=np.array(regreco9)
                            regreco10=np.array(regreco10)
                            
                            
                            
                            
                            
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor1[i]).T,(regrecor1[i]).T)
                                    covarregrecor1=np.append(covarregrecor1,C) 
                            covarregrecor1=np.resize(covarregrecor1,(64,nmaps*nmaps)) 
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor2[i]).T,(regrecor2[i]).T)
                                    covarregrecor2=np.append(covarregrecor2,np.resize(C,(1,nmaps*nmaps))) 
                            covarregrecor2=np.resize(covarregrecor2,(64,nmaps*nmaps)) 
                            
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor5[i]).T,(regrecor5[i]).T)
                                    covarregrecor5=np.append(covarregrecor5,np.resize(C,(1,nmaps*nmaps))) 
                            covarregrecor5=np.resize(covarregrecor5,(64,nmaps*nmaps)) 
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor6[i]).T,(regrecor6[i]).T)
                                    covarregrecor6=np.append(covarregrecor6,np.resize(C,(1,nmaps*nmaps))) 
                            covarregrecor6=np.resize(covarregrecor6,(64,nmaps*nmaps)) 
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor9[i]).T,(regrecor9[i]).T)
                                    covarregrecor9=np.append(covarregrecor9,np.resize(C,(1,nmaps*nmaps))) 
                            covarregrecor9=np.resize(covarregrecor9,(64,nmaps*nmaps)) 
                            
                            for i in range(64):
                                    C=cosine_similarity((regrecor10[i]).T,(regrecor10[i]).T)
                                    covarregrecor10=np.append(covarregrecor1,np.resize(C,(1,nmaps*nmaps))) 
                            covarregrecor10=np.resize(covarregrecor10,(64,nmaps*nmaps)) 
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            regrecor1=np.resize(regreco1,(64,2000*nmaps))
                            regrecor2=np.resize(regreco2,(64,2000*nmaps))
                            
                            regrecor5=np.resize(regreco5,(64,2000*nmaps))
                            regrecor6=np.resize(regreco6,(64,2000*nmaps))
                            
                            regrecor9=np.resize(regreco9,(64,2000*nmaps))
                            regrecor10=np.resize(regreco10,(64,2000*nmaps))  
                            
                            
                            
                            
                            
                            
                            

                            Xt=regrecor1[0:46,:]    
                            Xs=regrecor1[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=regrecor2[0:46,:]    
                            Xs=regrecor2[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=regrecor5[0:46,:]    
                            Xs=regrecor5[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=regrecor6[0:46,:]    
                            Xs=regrecor6[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=regrecor9[0:46,:]    
                            Xs=regrecor9[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=regrecor10[0:46,:]    
                            Xs=regrecor10[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Cpred_dtc.append(pred_dtc)
                            Cscore_dtc.append(score_dtc)

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            

                            Cpred_rf.append(pred_rf)
                            Cscore_rf.append(score_rf)

                            Cpred_svm.append(pred_svm)
                            Cscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            "covar based classification"
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor1[0:46,:]    
                            Xs=covarregrecor1[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor2[0:46,:]    
                            Xs=covarregrecor2[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor5[0:46,:]    
                            Xs=covarregrecor5[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor6[0:46,:]    
                            Xs=covarregrecor6[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor9[0:46,:]    
                            Xs=covarregrecor9[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            Xt=covarregrecor10[0:46,:]    
                            Xs=covarregrecor10[46:64,:]
                            Yt=score[0:46]   
                            Ys=score[46:64]
                                
                                
                                
                                
                                
                                
                                
                            from sklearn.tree import DecisionTreeClassifier
                            dtc=DecisionTreeClassifier()
                            
                            
                            dtc.fit(Xt,Yt)
                            
                            pred_dtc=dtc.predict(Xs)
                            score_dtc=dtc.score(Xs,Ys)
                                
                                
                                
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.datasets import make_classification
                            clf = RandomForestClassifier(max_depth=2, random_state=0)
                            clf.fit(Xt,Yt)
                            pred_rf=(clf.predict(Xs))
                            score_rf=(clf.score(Xs,Ys))
                            
                            
                            
                            from sklearn import svm
                            
                            clf = svm.SVC()
                            clf.fit(Xt,Yt)
                            pred_svm=(clf.predict(Xs))
                            score_svm=(clf.score(Xs,Ys))
                            
                            
                            
                            
                            
                            Ccorpred_dtc.append(pred_dtc)
                            Ccorscore_dtc.append(score_dtc)

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            

                            Ccorpred_rf.append(pred_rf)
                            Ccorscore_rf.append(score_rf)

                            Ccorpred_svm.append(pred_svm)
                            Ccorscore_svm.append(score_svm)
                            
                            
                            import pickle
                            
                            with open('Cpred_dtc.spydata', 'wb') as file:

                            # A new file will be created
                             pickle.dump(Cpred_dtc, file)
                             
                                                                                    
                             
                            with open('Cscore_dtc.spydata', 'wb') as file:
                        
                            # A new file will be created
                             pickle.dump(Cscore_dtc, file)
                             
                             
                             
                            with open('Cpred_rf.spydata', 'wb') as file:
                        
                            # A new file will be created
                             pickle.dump(Cpred_rf, file)
                             
                             
                             
                            with open('Cscore_rf.spydata', 'wb') as file:
                        
                            # A new file will be created
                             pickle.dump(Cscore_rf, file)
                             
                             
                             
                             
                            with open('Cpred_svm.spydata', 'wb') as file:
                        
                            # A new file will be created
                             pickle.dump(Cpred_svm, file)
                             
                             
                             
                            with open('Cpred_svm.spydata', 'wb') as file:
                        
                            # A new file will be created
                             pickle.dump(Cpred_svm, file)
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             
                             with open('Ccorpred_dtc.spydata', 'wb') as file:

                             # A new file will be Ccorreated
                              pickle.dump(Ccorpred_dtc, file)
                              
                              
                              
                              
                              
                             with open('Ccorscore_dtc.spydata', 'wb') as file:
                         
                             # A new file will be Ccorreated
                              pickle.dump(Ccorscore_dtc, file)
                              
                              
                              
                             with open('Ccorpred_rf.spydata', 'wb') as file:
                         
                             # A new file will be Ccorreated
                              pickle.dump(Ccorpred_rf, file)
                              
                              
                              
                             with open('Ccorscore_rf.spydata', 'wb') as file:
                         
                             # A new file will be Ccorreated
                              pickle.dump(Ccorscore_rf, file)
                              
                              
                              
                              
                             with open('Ccorpred_svm.spydata', 'wb') as file:
                         
                             # A new file will be Ccorreated
                              pickle.dump(Ccorpred_svm, file)
                              
                              
                              
                             with open('Ccorpred_svm.spydata', 'wb') as file:
                         
                             # A new file will be Ccorreated
                              pickle.dump(Ccorpred_svm, file)
    
    
    
    
    
    
    
    
    
    
    
                            Cpred_dtc1=np.array(Cpred_dtc)
                            Cscore_dtc1=np.array(Cscore_dtc)

                            Cpred_rf1=np.array(Cpred_rf)
                            Cscore_rf1=np.array(Cscore_rf)

                            Cpred_svm1=np.array(Cscore_svm)
                            Cscore_svm1=np.array(Cscore_svm)






                            Ccorpred_dtc1=np.array(Ccorpred_dtc)
                            Ccorscore_dtc1=np.array(Ccorscore_dtc)

                            Ccorpred_rf1=np.array(Ccorpred_rf)
                            Ccorscore_rf1=np.array(Ccorscore_rf)

                            Ccorpred_svm1=np.array(Ccorpred_svm)
                            Ccorscore_svm1=np.array(Ccorscore_svm)
    
    
    
    
    
    
                            from scipy.io import savemat
                            
                            mdic = {"Cpred_dtc": Cpred_dtc1}
                            
                            savemat("Cpred_dtc.mat", mdic)
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Cscore_dtc": Cscore_dtc1}
                            
                            savemat("Cscore_dtc.mat", mdic)
                            
                            from scipy.io import savemat
                            
                            mdic = {"Cpred_rf": Cpred_rf1}
                            
                            savemat("Cpred_rf.mat", mdic)
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Cscore_rf": Cscore_rf1}
                            
                            savemat("Cscore_rf.mat", mdic)
                            
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Cpred_svm": Cpred_svm1}
                            
                            savemat("Cpred_svm.mat", mdic)
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Cscore_svm": Cscore_svm1}
                            
                            savemat("Cscore_svm.mat", mdic)
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Ccorpred_dtc": Ccorpred_dtc1}
                            
                            savemat("Ccorpred_dtc.mat", mdic)
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Ccorscore_dtc": Ccorscore_dtc1}
                            
                            savemat("Ccorscore_dtc.mat", mdic)
                            
                            from scipy.io import savemat
                            
                            mdic = {"Ccorpred_rf": Ccorpred_rf1}
                            
                            savemat("Ccorpred_rf.mat", mdic)
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Ccorscore_rf": Ccorscore_rf1}
                            
                            savemat("Ccorscore_rf.mat", mdic)
                            
                            
                            
                            from scipy.io import savemat
                            
                            mdic = {"Ccorpred_svm": Ccorpred_svm1}
                            
                            savemat("Ccorpred_svm.mat", mdic)
                            
                            
                            from scipy.io import savemat
                           
                            mdic = {"Ccorscore_svm": Ccorscore_svm1}
                            
                            savemat("Ccorscore_svm.mat", mdic)
