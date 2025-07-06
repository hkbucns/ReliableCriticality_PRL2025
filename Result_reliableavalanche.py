import brian2 as br2
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import random
import pickle
import SpikingModel_Analysis

random.seed(2077)
np.random.seed(2077)

def Data_Generator_response_data():
    data_gen = SpikingModel_Analysis.SpikeTrain_and_network_Generator()
    model = SpikingModel_Analysis.SpikingNetworkModel()

    # Generate spike trains and connectivity

    connection = data_gen.generate_all_connections()

    # Run simulations
    data_save = {}
    data_save['connection'] = connection
    for signal in range(0, 3):

        spiketrain = data_gen.generate_poisson_spikes(frequency=25)
        ff_pattern = data_gen.generate_feedforward_pattern(signal_num=1)

        spikemat_signal = data_gen.generate_input_signal(spiketrain, ff_pattern[0])
        # data_save['signal_' + str(signal)] = spikemat_signal
        spiketime_list, spikeidx_list = [], []

        for trial in range(0, 100):
            data = model.run_simulation(tau_di= 8 * br2.ms,
                                        connection=connection, spikemat_signal=spikemat_signal)
            spiketime_list.append(data['e_t0'])
            spikeidx_list.append(data['e_i0'])
        data_save['response_st_' + str(signal)] = spiketime_list
        data_save['response_si_' + str(signal)] = spikeidx_list

    import os
    import pickle

    if not os.path.exists('example_data'):
        os.mkdir('example_data')
    savefile = open('example_data/network_response.pkl', 'wb')
    pickle.dump(data_save, savefile)
    savefile.close()

Data_Generator_response_data()

# Use the generator data to train a spike-based decoder and show the reliable avalanche.
savefile = open('example_data/network_response.pkl', 'rb')
data_save = pickle.load(savefile)
savefile.close()

tempotron = SpikingModel_Analysis.TempotronClassifier()
result = tempotron.train_classifier(SPIKETIME0 = data_save['response_st_0'], SPIKEINDEX0 = data_save['response_si_0'],
          SPIKETIME1 = data_save['response_st_2'], SPIKEINDEX1 = data_save['response_si_2'])

workflow = SpikingModel_Analysis.AvalancheAnalysisWorkflow(avalanche_threshold=50)
patterns = workflow.run_complete_analysis(data_save, num_conditions=3)
workflow.visualize_results(patterns)


