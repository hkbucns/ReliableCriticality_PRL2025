import brian2 as br2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

class SpikeTrain_and_network_Generator:
    """Generator for spike trains and neural activity patterns"""

    def __init__(self, seed=2077):
        # random.seed(seed)
        # np.random.seed(seed)
        self.dt = 0.01  # ms

    @staticmethod
    def time_idx_to_spiketrain(spiketime, spikeindex, netsize=800):
        """Convert time indices to spike train format"""
        spiketrain = []
        for neuron in range(netsize):
            stime = spiketime[np.where(spikeindex == neuron)[0]]
            if len(stime) == 0:
                spiketrain.append([])
            else:
                spiketrain.append(np.array(stime, dtype=np.float32))
        return spiketrain

    def generate_poisson_spikes(self, netsize=1000, frequency=25, duration=1000):
        """Generate Poisson spike trains"""
        netsize_e = int(netsize * 0.8)
        RunningTime = duration * br2.ms
        br2.defaultclock.dt = 0.01 * br2.ms

        P_e = br2.PoissonGroup(netsize, frequency * br2.Hz)
        mon = br2.SpikeMonitor(P_e)
        br2.run(RunningTime)

        spiketime, spikeidx = mon.t / br2.ms, mon.i + 0
        return self.time_idx_to_spiketrain(spiketime, spikeidx, netsize=netsize_e)

    def generate_feedforward_pattern(self, netsize=1000, overlap=0.9, signal_num=5):
        """Generate feedforward connectivity patterns with specified overlap"""
        network_e = int(netsize * 0.8)
        ConnectPoisson_list = []

        # Generate connection patterns for each signal
        for signal_idx in range(signal_num):
            ConnectPoisson = []
            for neuron in range(netsize):
                ConnectPoisson.append(random.sample(range(network_e), int(network_e * 0.2)))
            ConnectPoisson_list.append(ConnectPoisson)

        # Apply overlap constraint
        Sameidx = np.hstack([
            np.arange(0, int(network_e * overlap)),
            np.arange(network_e, network_e + int((netsize - network_e) * overlap))
        ])

        for signal_idx in range(signal_num):
            for neuron in range(netsize):
                if neuron in Sameidx:
                    ConnectPoisson_list[signal_idx][neuron] = ConnectPoisson_list[0][neuron]

        return ConnectPoisson_list

    def generate_input_signal(self, spiketrain, ff_pattern, netsize=1000, duration=1000):
        """Generate noise input matrix from spike trains and feedforward patterns"""
        spiketrain_signal = []

        for neuron in range(netsize):
            sampleindex = ff_pattern[neuron]
            signal = np.array([])

            for i in range(int(netsize * 0.8 * 0.2)):
                if i == 0:
                    signal = spiketrain[sampleindex[i]]
                else:
                    signal = np.hstack([signal, spiketrain[sampleindex[i]]])

            signal.sort()
            spiketrain_signal.append(signal)

        # Convert to spike matrix
        spikemat_signal = np.zeros((netsize, int(duration / self.dt)))
        for neuron in range(netsize):
            for spike in spiketrain_signal[neuron]:
                spikemat_signal[neuron, int(spike / self.dt)] = 1

        return spikemat_signal

    def generate_random_connectivity(self, source_num=800, target_num=800, p=0.2):
        """Generate random connectivity matrix"""
        connection_matrix = []
        for i in range(source_num):
            for j in range(target_num):
                if np.random.rand() < p:
                    connection_matrix.append([int(i), int(j)])
        return np.array(connection_matrix)

    def generate_all_connections(self):
        """Generate all connection matrices for E-E, E-I, I-E, I-I"""
        EECouMat = self.generate_random_connectivity(source_num=800, target_num=800)
        EICouMat = self.generate_random_connectivity(source_num=200, target_num=800)
        IECouMat = self.generate_random_connectivity(source_num=800, target_num=200)
        IICouMat = self.generate_random_connectivity(source_num=200, target_num=200)

        return {
            'EE': EECouMat,
            'EI': EICouMat,
            'IE': IECouMat,
            'II': IICouMat
        }
class SpikingNetworkModel:
    """Main spiking network model with fixed connectivity"""

    def __init__(self):
        self.setup_parameters()

    def setup_parameters(self):
        """Initialize model parameters"""
        # Time constants
        self.tau_e = 20 * br2.ms
        self.tau_i = 10 * br2.ms
        self.tau_de = 2 * br2.ms
        self.tau_r = 0.5 * br2.ms

        # Voltage parameters
        self.Ve_rest = -70
        self.Vi_rest = -70
        self.Ve_rev = 0
        self.Vi_rev = -70
        self.V_threshold = -50
        self.V_reset = -60

        # Network parameters
        self.N_total = 1000
        self.g = np.array([0.012, 0.024, 0.18, 0.31, 0.022, 0.040])

        #Simulation parameter
        self.time = 1000


    def run_simulation(self, tau_di=8*br2.ms, connection={}, spikemat_signal=np.zeros((10, 10))):
        tau_e,tau_i,tau_de,tau_r = self.tau_e,self.tau_i,self.tau_de,self.tau_r
        Ve_rest,Vi_rest,Ve_rev,Vi_rev,V_threshold,V_reset = (self.Ve_rest,self.Vi_rest,self.Ve_rev,
                                                            self.Vi_rev,self.V_threshold,self.V_reset)
        N_total,time = self.N_total,self.time

        """Run the main simulation with fixed network and input"""
        # Extract connection matrices
        EECouMat = connection['EE']
        EICouMat = connection['EI']
        IECouMat = connection['IE']
        IICouMat = connection['II']

        # Setup stimuli
        stimulus_e = br2.TimedArray(spikemat_signal[0:800, :].T, dt=0.01 * br2.ms)
        stimulus_i = br2.TimedArray(spikemat_signal[800:1000, :].T, dt=0.01 * br2.ms)

        del spikemat_signal

        # Setup timing
        RunningTime = self.time * br2.ms
        br2.defaultclock.dt = 0.01 * br2.ms

        # Conductance parameters
        g = self.g
        gee, gie, gei, gii, geo, gio = g

        # Setup equations

        eqs_e = '''
        dv/dt  = (Ve_rest-v)/tau_e + (Ve_rev-v)*g_e + (Vi_rev-v)*g_i : 1 (unless refractory)
        dg_e/dt = (-1/tau_r)*g_e + (1/(tau_r*tau_de))*x_e :Hz
        dx_e/dt = (-1/tau_de)*x_e + 1000*((geo*stimulus_e(t,i))/0.01) *Hz :1
        dg_i/dt = (-1/tau_r)*g_i + (1/(tau_r*tau_di))*x_i :Hz
        dx_i/dt = (-1/tau_di)*x_i :1
        '''
        eqs_i = '''
        dv/dt  = (Vi_rest-v)/tau_i + (Ve_rev-v)*g_e + (Vi_rev-v)*g_i : 1 (unless refractory)
        dg_e/dt = (-1/tau_r)*g_e + (1/(tau_r*tau_de))*x_e :Hz
        dx_e/dt = (-1/tau_de)*x_e + 1000*((gio*stimulus_i(t,i))/0.01)*Hz :1
        dg_i/dt = (-1/tau_r)*g_i + (1/(tau_r*tau_di))*x_i :Hz
        dx_i/dt = (-1/tau_di)*x_i :1
            '''


        # Create neuron groups
        P_e = br2.NeuronGroup(
            int(self.N_total * 0.8), eqs_e,
            threshold='v>V_threshold',
            reset='v = V_reset',
            refractory=2 * br2.ms,
            method='rk2'
        )

        P_i = br2.NeuronGroup(
            int(self.N_total * 0.2), eqs_i,
            threshold='v>V_threshold',
            reset='v = V_reset',
            refractory=1 * br2.ms,
            method='rk2'
        )

        # Initialize states
        P_e.v = 'V_reset + rand() * (V_threshold - V_reset)'
        P_e.g_e = '200*rand()*Hz'
        P_e.x_e = '0.4*rand()'
        P_e.g_i = '200*rand()*Hz'
        P_e.x_i = '0.4*rand()'

        P_i.v = 'V_reset + rand() * (V_threshold - V_reset)'
        P_i.g_e = '200*rand()*Hz'
        P_i.x_e = '0.4*rand()'
        P_i.g_i = '200*rand()*Hz'
        P_i.x_i = '0.4*rand()'

        # Create synapses
        Cee = br2.Synapses(P_e, P_e, on_pre=f'x_e += {gee}')
        Cie = br2.Synapses(P_e, P_i, on_pre=f'x_e += {gie}')
        Cei = br2.Synapses(P_i, P_e, on_pre=f'x_i += {gei}')
        Cii = br2.Synapses(P_i, P_i, on_pre=f'x_i += {gii}')

        # Connect synapses
        Cee.connect(i=EECouMat[:, 0], j=EECouMat[:, 1])
        Cie.connect(i=IECouMat[:, 0], j=IECouMat[:, 1])
        Cei.connect(i=EICouMat[:, 0], j=EICouMat[:, 1])
        Cii.connect(i=IICouMat[:, 0], j=IICouMat[:, 1])

        # Setup monitors
        e_mon = br2.SpikeMonitor(P_e)
        # i_mon = br2.SpikeMonitor(P_i)

        # Run simulation
        br2.run(RunningTime, report='text')

        return {
            'e_t0': e_mon.t / br2.ms,
            'e_i0': e_mon.i + 0
        }
class AvalancheAnalyzer:
    """Analyzer for neural avalanche detection and statistics"""

    def __init__(self, cut_size=300000, avalanche_threshold=50):
        self.cut_size = cut_size
        self.avalanche_threshold = avalanche_threshold

    def analyze_single_trial(self, spiketime, spikeidx):
        """Analyze avalanche statistics for a single trial"""
        # Calculate average inter-spike interval for time binning
        ISI = spiketime[1:] - spiketime[0:-1]
        ISI_ave = np.mean(ISI)

        # Split data into chunks for processing
        cutnum = int(len(spiketime) / self.cut_size)
        Sizelist_bin, Sizenidx_bin = [], []

        # Process each chunk
        for cutindex in range(cutnum + 1):
            beginindex = cutindex * self.cut_size
            endindex = min((cutindex + 1) * self.cut_size, len(spiketime))

            if beginindex >= len(spiketime):
                break

            spiketime_temp = list(spiketime[beginindex:endindex] - spiketime[beginindex])
            spikeindex_temp = list(spikeidx[beginindex:endindex])

            # Bin spikes into discrete time windows
            max_time = max(spiketime_temp) if spiketime_temp else 0
            for i in range(int(max_time / ISI_ave)):
                t_end = (i + 1) * ISI_ave
                neuron_indices = []
                spike_count = 0

                while spiketime_temp and spiketime_temp[0] < t_end:
                    neuron_indices.append(spikeindex_temp.pop(0))
                    spiketime_temp.pop(0)
                    spike_count += 1

                Sizelist_bin.append(spike_count)
                Sizenidx_bin.append(neuron_indices)

        # Calculate avalanche durations and sizes
        avalanche_sizes, avalanche_lengths, avalanche_indices = [], [], []
        avalanche_times = []
        i = 0

        while i < len(Sizelist_bin):
            if Sizelist_bin[i] != 0:
                length = 1
                # Find continuous non-zero activity
                while (i + length < len(Sizelist_bin) and
                       Sizelist_bin[i + length] != 0):
                    length += 1

                avalanche_lengths.append(length)
                avalanche_sizes.append(sum(Sizelist_bin[i:i + length]))
                avalanche_indices.append(np.hstack(Sizenidx_bin[i:i + length]))
                avalanche_times.append(i)

                i += length
            else:
                i += 1

        # Calculate avalanche timing
        avalanche_times = (np.array(avalanche_times) +
                           np.array(avalanche_lengths) / 2) * ISI_ave

        # Filter large avalanches
        if avalanche_sizes:
            avalanche_sizes[0] = 0  # Remove first avalanche artifact

        large_avalanche_idx = np.where(np.array(avalanche_sizes) > self.avalanche_threshold)[0]
        large_avalanche_times = avalanche_times[large_avalanche_idx]

        return large_avalanche_times, avalanche_indices, avalanche_sizes, avalanche_times

    def analyze_multiple_trials(self, spiketime_list, spikeidx_list):
        """Analyze avalanche statistics across multiple trials"""
        results = {
            'large_avalanche_times': [],
            'avalanche_indices': [],
            'avalanche_sizes': [],
            'all_avalanche_times': []
        }

        for trial in range(len(spiketime_list)):
            large_times, indices, sizes, all_times = self.analyze_single_trial(
                spiketime_list[trial], spikeidx_list[trial]
            )

            results['large_avalanche_times'].append(large_times)
            results['avalanche_indices'].append(indices)
            results['avalanche_sizes'].append(sizes)
            results['all_avalanche_times'].append(all_times)

        return results
class ReliableAvalancheDetector:
    # Detector for reliable avalanches across trials

    def __init__(self, reliability_threshold=0.8, time_bin_size=5):
        self.reliability_threshold = reliability_threshold
        self.time_bin_size = time_bin_size
        self.time_shift_bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        self.avalanche_threshold = 50

    def find_reliable_avalanches(self, avalanche_results, max_time=1000, num_trials=100):
        # Find avalanches that occur reliably across trials
        large_avalanche_times = avalanche_results['large_avalanche_times']

        # Create binary matrix for avalanche occurrence
        time_bins = int(max_time / self.time_bin_size)
        avalanche_matrix = np.zeros((num_trials, time_bins))

        for trial in range(num_trials):
            for avalanche_time in large_avalanche_times[trial]:
                bin_idx = int(avalanche_time / self.time_bin_size)
                if bin_idx < time_bins:
                    avalanche_matrix[trial, bin_idx] = 1

        # Calculate reliability across trials
        avalanche_reliability = np.mean(avalanche_matrix, axis=0)

        # Find highly reliable avalanches
        reliable_avalanche_bins = np.where(avalanche_reliability > self.reliability_threshold)[0]

        return reliable_avalanche_bins, avalanche_reliability

    def extract_neuron_patterns(self, avalanche_results, reliable_bins):
        # Extract neuron activation patterns for reliable avalanches
        avalanche_indices = avalanche_results['avalanche_indices']
        avalanche_sizes = avalanche_results['avalanche_sizes']
        all_avalanche_times = avalanche_results['all_avalanche_times']

        neuron_patterns = []
        pattern_times = []
        temporal_reliability = []
        spatial_reliability = []

        for bin_idx in reliable_bins:

            best_pattern = None
            best_shift = 0
            best_temporal_rel = 0
            best_spatial_rel = 0

            # Try different time shifts to find optimal alignment
            for shift in self.time_shift_bins:
                precise_time = bin_idx * self.time_bin_size + shift
                pattern = self._extract_pattern_at_time(
                    precise_time, avalanche_indices, avalanche_sizes, all_avalanche_times
                )

                temporal_rel, spatial_rel = self._calculate_reliability(pattern)

                if temporal_rel > best_temporal_rel:
                    best_pattern = pattern
                    best_shift = shift
                    best_temporal_rel = temporal_rel
                    best_spatial_rel = spatial_rel

            neuron_patterns.append(best_pattern)
            pattern_times.append(bin_idx * self.time_bin_size + best_shift)
            temporal_reliability.append(best_temporal_rel)
            spatial_reliability.append(best_spatial_rel)

        return neuron_patterns, pattern_times, temporal_reliability, spatial_reliability

    def _extract_pattern_at_time(self, target_time, avalanche_indices, avalanche_sizes, all_avalanche_times):
        # Extract neuron activation pattern at specific time
        pattern = np.zeros((800, 100))  # 800 neurons, 100 trials

        for trial in range(100):
            if trial >= len(avalanche_sizes):
                continue

            sizes = np.array(avalanche_sizes[trial])
            times = all_avalanche_times[trial]
            indices = avalanche_indices[trial]

            # Find large avalanches near target time
            large_idx = np.where(sizes > self.avalanche_threshold)[0]
            if len(large_idx) == 0:
                continue

            large_times = times[large_idx]
            nearby_idx = np.where((large_times > target_time - 1) &
                                  (large_times < target_time + 6))[0]

            if len(nearby_idx) > 0:
                # Select largest avalanche if multiple candidates
                candidate_sizes = [len(indices[large_idx[idx]]) for idx in nearby_idx]
                best_candidate = nearby_idx[np.argmax(candidate_sizes)]
                neuron_list = indices[large_idx[best_candidate]]

                for neuron in neuron_list:
                    if neuron < 800:  # Ensure valid neuron index
                        pattern[neuron, trial] = 1

        return pattern

    def _calculate_reliability(self, pattern):
        # Calculate temporal and spatial reliability of a pattern
        # Temporal reliability: fraction of trials with non-zero activity
        active_trials = np.sum(pattern, axis=0) > 0
        temporal_reliability = np.sum(active_trials) / len(active_trials)

        # Spatial reliability: correlation between active trials
        if np.sum(active_trials) <= 1:
            spatial_reliability = 0
        else:
            active_pattern = pattern[:, active_trials]
            if active_pattern.shape[1] > 1:
                correlations = np.corrcoef(active_pattern.T)
                n = len(correlations)
                spatial_reliability = (np.sum(correlations) - n) / (n * (n - 1))
            else:
                spatial_reliability = 0

        return temporal_reliability, spatial_reliability
class PatternDimensionalityReducer:
    # Dimensionality reduction for neural patterns

    def __init__(self, method='pca'):
        self.method = method
        self.color_palette = [
            'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:pink',
            'tab:gray', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan',
            'black', 'yellow', 'peru', 'navy', 'lavender',
            'lime', 'violet', 'teal', 'crimson', 'rosybrown'
        ]

    def reduce_dimensions(self, pattern_list):
        # Reduce dimensionality of neural patterns
        # Combine all patterns
        combined_patterns = []
        pattern_lengths = []

        for i, pattern in enumerate(pattern_list):
            if len(pattern) > 0:
                # Remove trials with no activity
                active_trials = np.where(np.sum(pattern, axis=0) != 0)[0]
                if len(active_trials) > 0:
                    pattern_data = pattern[:, active_trials]
                    if i == 0:
                        combined_patterns = pattern_data
                    else:
                        combined_patterns = np.hstack([combined_patterns, pattern_data])
                    pattern_lengths.append(combined_patterns.shape[1])
                else:
                    pattern_lengths.append(0 if i == 0 else pattern_lengths[-1])
            else:
                pattern_lengths.append(0 if i == 0 else pattern_lengths[-1])

        if len(combined_patterns) == 0:
            return None, [], []

        # Normalize and perform dimensionality reduction
        combined_patterns = combined_patterns.T
        combined_patterns = normalize(combined_patterns)

        if self.method == 'pca':
            pca = PCA(n_components=min(3, combined_patterns.shape[1]))
            reduced_data = pca.fit_transform(combined_patterns)
        else:  # Original covariance method
            cov_matrix = np.dot(combined_patterns.T, combined_patterns)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            reduced_data = np.dot(combined_patterns, eigenvectors)

        # Generate colors for different patterns
        colors = self._generate_colors(pattern_lengths)

        return reduced_data, colors, pattern_lengths

    def _generate_colors(self, pattern_lengths):
        """Generate colors for different pattern groups"""
        colors = []
        for i in range(len(pattern_lengths)):
            if i == 0:
                length = pattern_lengths[i]
            else:
                length = pattern_lengths[i] - pattern_lengths[i - 1]

            color = self.color_palette[i % len(self.color_palette)]
            colors.extend([color] * length)

        return colors

class AvalancheAnalysisWorkflow:
    # Complete workflow for avalanche analysis

    def __init__(self, avalanche_threshold=50):
        self.analyzer = AvalancheAnalyzer(avalanche_threshold=avalanche_threshold)
        self.detector = ReliableAvalancheDetector()
        self.reducer = PatternDimensionalityReducer()

    def run_complete_analysis(self, data_dict, num_conditions=3):
        # Run complete avalanche analysis workflow
        all_patterns = []

        for condition_idx in range(num_conditions):
            print(f'Processing condition {condition_idx + 1}/{num_conditions}')

            # Extract spike data for current condition
            spiketime_key = f'response_st_{condition_idx}'
            spikeidx_key = f'response_si_{condition_idx}'

            if spiketime_key not in data_dict or spikeidx_key not in data_dict:
                print(f'Warning: Data for condition {condition_idx} not found')
                continue

            # Analyze avalanches
            avalanche_results = self.analyzer.analyze_multiple_trials(
                data_dict[spiketime_key], data_dict[spikeidx_key]
            )

            # Find reliable avalanches
            reliable_bins, reliability = self.detector.find_reliable_avalanches(avalanche_results)

            if len(reliable_bins) == 0:
                print(f'No reliable avalanches found for condition {condition_idx}')
                all_patterns.append(np.array([]))
                continue

            # Extract neuron patterns
            patterns, times, temporal_rel, spatial_rel = self.detector.extract_neuron_patterns(
                avalanche_results, reliable_bins
            )

            # Select most spatially reliable pattern
            if len(spatial_rel) > 0:
                best_pattern_idx = np.argmax(spatial_rel)
                best_pattern = patterns[best_pattern_idx]
                all_patterns.append(best_pattern)
                print(f'Best avalanche spatial reliability: {spatial_rel[best_pattern_idx]:.3f}')
            else:
                all_patterns.append(np.array([]))

        return all_patterns

    def visualize_results(self, patterns):
        # Visualize dimensionality reduction results
        reduced_data, colors, lengths = self.reducer.reduce_dimensions(patterns)

        if reduced_data is None:
            print("No data to visualize")
            return

        # Create visualization
        plt.figure(figsize=(3, 2.5))

        # 2D visualization
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors, alpha=0.7)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('2D Pattern Visualization')

        plt.tight_layout()
        plt.show()

        return reduced_data, colors, lengths
class TempotronClassifier:
    """Tempotron-based classification algorithm"""

    def __init__(self):
        self.tau_r = 0.5
        self.tau_de = 2
        self.visualization=True

    def synaptic_activation(self, delta_t):
        """Calculate synaptic activation function"""
        delta_t[delta_t < 0] = 0
        return -np.exp(-delta_t / self.tau_r) + np.exp(-delta_t / self.tau_de)

    def calculate_voltage_max(self, W, spiketrain):
        """Calculate maximum voltage and its timing"""
        V = np.zeros(900)
        t = np.linspace(100, 999, 900)

        for neuron in range(800):
            for tspike in spiketrain[neuron]:
                V = V + W[neuron] * self.synaptic_activation(t - tspike)

        V_max = np.max(V)
        t_max = t[np.argmax(V)]
        return t_max, V_max, V

    def calculate_weight_update(self, t_max, spiketrain):
        """Calculate weight update based on spike timing"""
        delta_W = np.zeros(800)

        for neuron in range(800):
            spiketimelist = np.array(spiketrain[neuron])
            spiketimelist[spiketimelist > t_max] = t_max
            delta_W[neuron] = np.sum(self.synaptic_activation(t_max - spiketimelist))

        return delta_W

    def train_classifier(self, SPIKETIME0, SPIKEINDEX0, SPIKETIME1, SPIKEINDEX1):
        """Train Tempotron classifier"""
        # Training parameters
        trail_train = np.arange(0, 20)
        Lambda = 0.5  # learning rate
        V_threshold = Lambda * 100
        Maxstep = 100

        # Prepare training data
        method = SpikeTrain_and_network_Generator()

        spiketrain0list, spiketrain1list = [], []
        for trail in trail_train:
            spiketrain0list.append(method.time_idx_to_spiketrain(SPIKETIME0[trail], SPIKEINDEX0[trail]))
            spiketrain1list.append(method.time_idx_to_spiketrain(SPIKETIME1[trail], SPIKEINDEX1[trail]))

        # Initialize weights
        W = np.abs(np.random.normal(0, 0.1, 800))
        W_rec = np.zeros((800, Maxstep))

        # Training loop
        for i in range(Maxstep):
            delta_W = 0
            V_max_poslist, V_max_neglist = [], []

            for trail in trail_train:
                # Calculate maximum voltages
                t_max_pos, V_max_pos0, _ = self.calculate_voltage_max(W, spiketrain0list[trail])
                t_max_neg, V_max_neg0, _ = self.calculate_voltage_max(W, spiketrain1list[trail])

                # Update weights
                delta_W += Lambda * (
                        self.calculate_weight_update(t_max_pos, spiketrain0list[trail]) -
                        self.calculate_weight_update(t_max_neg, spiketrain1list[trail])
                ) / len(trail_train)

                V_max_poslist.append(V_max_pos0)
                V_max_neglist.append(V_max_neg0)

            print(f'Step {i}; V_max_pos: {min(V_max_poslist)}; V_max_neg: {max(V_max_neglist)}')

            W = W + delta_W
            W_rec[:, i] = W

            # Check convergence conditions
            condition_pass = (max(V_max_neglist) < V_threshold) and (min(V_max_poslist) > V_threshold)
            condition_quit = (max(V_max_neglist) > V_threshold + 5) and (min(V_max_poslist) > V_threshold + 5)

            if condition_pass or condition_quit:
                break

        # Test phase
        V_max_poslist, V_max_neglist = [], []
        t_tempotron_work = np.zeros(100)

        for trail in range(100):
            t_tempotron_work[trail], V_max_pos_prd, _ = self.calculate_voltage_max(
                W, method.time_idx_to_spiketrain(SPIKETIME0[trail], SPIKEINDEX0[trail])
            )
            V_max_poslist.append(V_max_pos_prd)

            _, V_max_neg_prd, _ = self.calculate_voltage_max(
                W, method.time_idx_to_spiketrain(SPIKETIME1[trail], SPIKEINDEX1[trail])
            )
            V_max_neglist.append(V_max_neg_prd)

        if self.visualization:
            V_max_neglist, V_max_poslist = np.array(V_max_neglist), np.array(V_max_poslist)
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.hist(W, bins=40, weights=np.ones(len(W)) / len(W))
            plt.xlabel('W')
            plt.ylabel('Prob.')
            plt.subplot(1, 2, 2)
            plt.plot(range(0, 100), V_max_poslist, '.', color='red')
            plt.plot(len(trail_train) * np.ones(100), np.linspace(min(V_max_neglist), max(V_max_poslist), 100), '-',
                     color='black')
            plt.plot(range(0, 100), 50 * np.ones(100), '-', color='black')
            plt.plot(range(0, 100), V_max_neglist, '.', color='blue')
            plt.xlabel('Trial')
            plt.ylabel('Maximal Voltage')
            plt.tight_layout()
            plt.show()

        V_max_neglist, V_max_poslist = V_max_neglist[20:], V_max_poslist[20:]
        CR = (len(np.where(V_max_neglist > V_threshold)[0]) + len(np.where(V_max_poslist < V_threshold)[0])) / 160

        return {'acc':CR,'weight':W,'step':i,'readout_timepoint':t_tempotron_work}

