import brian2 as br2
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import random

random.seed(2077)
np.random.seed(2077)

def AvalanceSizeAndLength(spiketime,cutsize = 300000):
    # Use the average interspikeintervel as time bin
    spiketime = spiketime[spiketime > 200]-200
    ISI = spiketime[1:] - spiketime[0:-1]
    ISI_ave = np.mean(ISI)
    # print('spike_num:'+str(len(spiketime))+' aveISI:'+str(ISI_ave))
    cutnum = int(len(spiketime)/cutsize)
    lastsize = len(spiketime)%cutsize

    Sizelist_bin = []
    for cutindex in range(0,cutnum+1):
        beginindex = cutindex*cutsize
        endindex =np.min([(cutindex+1)*cutsize,len(spiketime)])
        spiketime_tem = np.ndarray.tolist(spiketime[beginindex:endindex]-spiketime[beginindex])
        # Pool the spike train into discrete bin
        for i in range(0, int((max(spiketime_tem))/ISI_ave) - 1):
            t_end = (i + 1) * ISI_ave;
            j = 0
            while spiketime_tem[0] < t_end:
                del (spiketime_tem[0])
                j = j + 1
            Sizelist_bin.append(j)
    # Calculate the avalanche size and duration
    SizeList = [];  DurationList = []
    ii = 0

    for i in range(0, len(Sizelist_bin)):  # len(a)-1
        # print([i,Sizelist_bin[i]])
        if i >= ii:
            if Sizelist_bin[i] != 0:
                length = 1
                if i < len(Sizelist_bin) - 1:
                    while (Sizelist_bin[i + length] != 0):
                        length = length + 1
                        if i + length >= len(Sizelist_bin):
                            break
                DurationList.append(length)
                SizeList.append(sum(Sizelist_bin[i:i + length]))
                ii = i + length

    return (DurationList, SizeList)

def powerlawplot(Duration,Size,givedura=0):
    binedge0_L, prob0_L = powerlaw.pdf(Duration, linear_bins=True)
    binedge0_S, prob0_S = powerlaw.pdf(Size, linear_bins=True)

    fit_size = powerlaw.Fit(Size, discrete=True, xmin=(0, 20), xmax=np.max(Size))
    tau_size = fit_size.power_law.alpha
    fit_sizex = binedge0_S[1:][np.where(binedge0_S[1:]>=np.min(fit_size.data))]

    fit_sizey = np.power(fit_sizex,-tau_size)
    fit_sizey = fit_sizey*prob0_S[np.where(binedge0_S[1:]==np.min(fit_size.data))]/fit_sizey[0]

    fit_duration = powerlaw.Fit(Duration, discrete=True, xmin=(0, 10), xmax=np.max(Duration))
    tau_duration = fit_duration.power_law.alpha

    if givedura==1:
        tau_duration=2.0

    fit_durationx = binedge0_L[1:][np.where(binedge0_L[1:] >= np.min(fit_duration.data))]
    fit_durationy = np.power(fit_durationx, -tau_duration)
    fit_durationy = fit_durationy * prob0_L[np.where(binedge0_L[1:]== np.min(fit_duration.data))] / fit_durationy[0]

    return [tau_size,tau_duration],[binedge0_S[1:],prob0_S],[binedge0_L[1:],prob0_L],[fit_sizex,fit_sizey],[fit_durationx,fit_durationy]

def Spikingmodel(tau_di=[8*br2.ms],time = 1000*20,N_total=1000):

    g_size=1/np.sqrt(N_total/1000); bf=25
    node = 1
    p = 0.2*np.ones([node]); backgroundfre=bf*np.ones([node])

    [RunningTime,br2.defaultclock.dt] = [time*br2.ms,0.01*br2.ms]

    [tau_e,tau_i,tau_de,tau_r] = [20 * br2.ms,10*br2.ms,2*br2.ms,0.5*br2.ms]
    [Ve_rest,Vi_rest,Ve_rev,Vi_rev,V_threshold,V_reset]=[-70,-70,0,-70,-50,-60]

    g = np.array([0.012, 0.024, 0.18, 0.31,0.022, 0.040])*g_size
    [gee, gie, gei, gii, geo, gio] = [g[0],g[1],g[2],g[3],g[4],g[5]]


    ## Dynamic model
    eqs_e = '''
    dv/dt  = (Ve_rest-v)/tau_e + (Ve_rev-v)*g_e + (Vi_rev-v)*g_i : 1 (unless refractory)
    dg_e/dt = (-1/tau_r)*g_e + (1/(tau_r*tau_de))*x_e :Hz
    dx_e/dt = (-1/tau_de)*x_e :1
    dg_i/dt = (-1/tau_r)*g_i + (1/(tau_r*tau_di))*x_i :Hz
    dx_i/dt = (-1/tau_di)*x_i :1
    '''
    eqs_i = '''
    dv/dt  = (Vi_rest-v)/tau_i + (Ve_rev-v)*g_e + (Vi_rev-v)*g_i : 1 (unless refractory)
    dg_e/dt = (-1/tau_r)*g_e + (1/(tau_r*tau_de))*x_e :Hz
    dx_e/dt = (-1/tau_de)*x_e :1
    dg_i/dt = (-1/tau_r)*g_i + (1/(tau_r*tau_di))*x_i :Hz
    dx_i/dt = (-1/tau_di)*x_i :1
    '''
    # Local population dynamic
    for i in range(0,node):
        # Define excitation population dynamic
        locals()['P_e'+str(i)] = br2.NeuronGroup(N_total * 0.8, eqs_e, threshold='v>V_threshold',
                                reset='v = V_reset',refractory=2 * br2.ms, method='rk2',namespace={'tau_di': tau_di[i]})
        locals()['P_e'+str(i)].v = 'V_reset + rand() * (V_threshold - V_reset)'
        [locals()['P_e'+str(i)].g_e,locals()['P_e'+str(i)].x_e,locals()['P_e'+str(i)].g_i,locals()['P_e'+str(i)].x_i] = \
            ['200*rand()*Hz','0.4*rand()','200*rand()*Hz','0.4*rand()']
        # Define inhibition population dynamic
        locals()['P_i'+str(i)] = br2.NeuronGroup(N_total * 0.2, eqs_i, threshold='v>V_threshold',
                                reset='v = V_reset',refractory=1 * br2.ms, method='rk2',namespace={'tau_di': tau_di[i]})
        locals()['P_i'+str(i)].v = 'V_reset + rand() * (V_threshold - V_reset)'
        [locals()['P_i'+str(i)].g_e,locals()['P_i'+str(i)].x_e,locals()['P_i'+str(i)].g_i,locals()['P_i'+str(i)].x_i] = \
            ['200*rand()*Hz','0.4*rand()','200*rand()*Hz','0.4*rand()']

        # Define intra-area coupling
        [locals()['gee' + str(i)],locals()['gie' + str(i)],locals()['gei' + str(i)],locals()['gii' + str(i)]]\
            =[gee, gie, gei, gii]
        locals()['Cee'+str(i)] = br2.Synapses(source=locals()['P_e'+str(i)], target=locals()['P_e'+str(i)], on_pre='x_e += gee'+str(i))
        locals()['Cie'+str(i)] = br2.Synapses(source=locals()['P_e'+str(i)], target=locals()['P_i'+str(i)], on_pre='x_e += gie'+str(i))
        locals()['Cei'+str(i)] = br2.Synapses(source=locals()['P_i'+str(i)], target=locals()['P_e'+str(i)], on_pre='x_i += gei'+str(i))
        locals()['Cii'+str(i)] = br2.Synapses(source=locals()['P_i'+str(i)], target=locals()['P_i'+str(i)], on_pre='x_i += gii'+str(i))
        locals()['Cee' + str(i)].connect(p=p[i]);locals()['Cie' + str(i)].connect(p=p[i])
        locals()['Cei' + str(i)].connect(p=p[i]);locals()['Cii' + str(i)].connect(p=p[i])

        #Background input
        locals()['Poisson_e'+str(i)] = br2.PoissonGroup(int(N_total*0.8), backgroundfre[i]*br2.Hz)
        locals()['S_input_e'+str(i)] = br2.Synapses(locals()['Poisson_e'+str(i)], locals()['P_e'+str(i)], on_pre='x_e += geo')
        locals()['S_input_i'+str(i)] = br2.Synapses(locals()['Poisson_e'+str(i)], locals()['P_i'+str(i)], on_pre='x_e += gio')
        locals()['S_input_e' + str(i)].connect(p=p[i]); locals()['S_input_i'+str(i)].connect(p=p[i])


    # Monitor
    for i in range(0,node):
        locals()['e_mon'+str(i)] = br2.SpikeMonitor(locals()['P_e'+str(i)])
        locals()['i_mon'+str(i)] = br2.SpikeMonitor(locals()['P_i'+str(i)])
        locals()['evolt_mon'+str(i)] = br2.StateMonitor(locals()['P_e'+str(i)], 'v', record=True, dt= 1* br2.ms)
        locals()['ivolt_mon'+str(i)] = br2.StateMonitor(locals()['P_i'+str(i)], 'v', record=True, dt= 1* br2.ms)


    # Run_model
    br2.run(RunningTime, report='text')
    # SpikeStast
    for i in range(0,node):
        if i == 0:
            frlist=[];Velist=[];Vilist=[]
            sigmalist=np.zeros([node,2]); fr_Vlist=np.zeros([node,2])
            Ve=[]
        espike=locals()['e_mon'+str(i)].t/br2.ms; espike = np.delete(espike, np.where(espike < 200))
        ispike=locals()['i_mon'+str(i)].t/br2.ms; ispike = np.delete(ispike, np.where(ispike < 200))
        [evolt,ivolt] = [np.float16(locals()['evolt_mon'+str(i)].v), np.float16(locals()['ivolt_mon'+str(i)].v)]
        [evolt_mean, ivolt_mean] = [np.mean(evolt[:, 200:]),np.mean(ivolt[:, 200:])]

        voltage_e = np.mean(evolt, 0);voltage_i = np.mean(ivolt, 0)
        Velist.append(list(voltage_e));Vilist.append(list(voltage_i))
        Ve.append(evolt[:, 200:])

        [efr,ifr]=[len(espike) / (N_total * 0.8 * (RunningTime / br2.ms - 200)),
                                 len(ispike) / (N_total * 0.2 * (RunningTime / br2.ms - 200))]
        [sigma_e,sigma_i]= [((V_threshold - evolt_mean) / (np.log(efr ** (-1) - 1))) * (np.pi / np.sqrt(3)),
                                                ((V_threshold - ivolt_mean) / (np.log(ifr ** (-1) - 1))) * (np.pi / np.sqrt(3))]
        sigmalist[i,0]=sigma_e;sigmalist[i,1]=sigma_i;fr_Vlist[i,0]=efr*1000;fr_Vlist[i,1]=evolt_mean


    data={'e_t0':locals()['e_mon'+str(0)].t/br2.ms, 'e_i0':locals()['e_mon'+str(0)].i+0,
          'i_t0':locals()['i_mon'+str(0)].t/br2.ms, 'i_i0':locals()['i_mon'+str(0)].i+0, 'Velist':np.array(Velist),'Ve':Ve,'sigma':sigmalist}
    return data

data = Spikingmodel(tau_di=[8*br2.ms],time = 1000*10)

Duration_list,Size_list = AvalanceSizeAndLength(data['e_t0'])

[tau_size_BP,tau_duration_BP],[binS,prob0_S],[binD,prob0_D],[fit_sizex,fit_sizey],[fit_durationx,fit_durationy] =\
    powerlawplot(Duration_list,Size_list,givedura=0)

if len(np.where(fit_durationy<np.min(prob0_D[np.where(prob0_D>0)]))[0])==0:
    Durationstop = len(fit_durationy)
else:
    Durationstop = np.where(fit_durationy<np.min(prob0_D[np.where(prob0_D>0)]))[0][0]
Sizestop = np.where(fit_sizey<np.min(prob0_S[np.where(prob0_S>0)]))[0][0]

plt.figure(figsize=(8,2))
plt.subplot(1,3,1)
plt.plot(data['e_t0'],data['e_i0'],'.',markersize=0.5,color='tab:red')
plt.plot(data['i_t0'],data['i_i0']+800,'.',markersize=0.5,color='tab:blue')
plt.xlim([200,500])
plt.ylim([0,1000])
plt.xlabel('Time (ms)')
plt.subplot(1,3,2)
plt.loglog(binD, prob0_D,'.',markersize=3.0,color='tab:blue',alpha=0.8)
plt.loglog(fit_durationx[0:Durationstop], fit_durationy[0:Durationstop],color='k',alpha=0.8)
plt.xlabel('Avalanche Duration')
plt.ylabel('Prob.')
plt.subplot(1,3,3)
plt.loglog(binS, prob0_S,'.',markersize=3.0,color='tab:red',alpha=0.8)
plt.loglog(fit_sizex[0:Sizestop], fit_sizey[0:Sizestop],color='k',alpha=0.8)
plt.xlabel('Avalanche Size')
plt.tight_layout()
plt.show()
