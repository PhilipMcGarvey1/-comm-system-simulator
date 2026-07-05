&#x09;Motivation for Project



This project is an effort to learn more about how communication systems function from start to finish as a summer portfolio electrical engineering project.



It looks to build a simulation of the basic principles of both analog and digital communication including modulation, noise, demodulation, filtering, SNR calculations, FFT, and BER.



The idea was to connect knowledge gained in courses on signals and systems and signal processing to a practical Python simulator.





&#x09;First Thing I Built



I started with organizing the project structure where I created separate folders for:



\- signal generation

\- modulation

\- demodulation

\- analysis

\- visualization

\- results



Doing so, I aimed to make the project scalable rather than put all my code into one single Python file.





&#x09;Key Engineering Concepts I Applied



With this project, I learned how to apply:  

\- generation of sampled time-domain signals

\- creation of multi tone baseband signals

\- modulation of signals through AM and DSB-SC

\- addition of channel noise through AWGN

\- demodulation of signals through envelope and coherent demodulation

\- recovery of baseband information through low-pass filtering

\- calculation of SNR and error metrics

\- use of FFTs to view signals in the frequency domain

\- generation of random bits

\- mapping of bits into BPSK symbols

\- demodulation of BPSK signals

\- calculation of BER at various Eb/N0 values

\- creation of an interactive Streamlit dashboard



&#x09;Design Choices



&#x09;Sampling Frequency



I set the sampling frequency to a default value of 100 kHz to ensure that the message frequencies and carrier frequency were represented clearly.



&#x09;Message Signal



A multi-tone message signal consisting of frequencies of 500 Hz, 1000 Hz, and 2000 Hz was chosen for easier verification in the FFT outputs.     





&#x09;Carrier Frequency



I used 10 kHz for the carrier frequency as it was much higher than the message frequencies but still relatively easy to plot and simulate.



&#x20;	Low-Pass Filter



I used 3 kHz cutoff for the low-pass filter since the highest frequency in the message tones was 2 kHz. It allows the receiver to retain the message and filter out any higher frequencies.



&#x09;BPSK



I implemented the BPSK as I needed the project to contain a digital communication system as well as analog AM modulation.





&#x09;Problems Encountered



Among the major problems I encountered were:



\- Python configuration in VS Code

\- Issues with importing libraries between different folders

\- Why `\_\_init\_\_.py` files are necessary

\- Proper implementation of imports from Streamlit to project modules

\- Difference between SNR and Eb/N0

\- Correct decreasing behavior of BER graph with the increase in Eb/N0

\- Need of filtering in demodulation



&#x09;Insights Gained



What I have gained from this project the most is the realization that communication systems consist of several steps. Complete system requires the following chain:



signal source -> modulation -> channel -> receiver -> analysis   

