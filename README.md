End-to-End Communication System Simulator



This is a Python simulator for analog and digital communication links. You'll find everything bundled together — from generating signals to watching them get noisy and then trying to clean them back up.



What's inside:



\- Signal generation

\- AM modulation and DSB-SC modulation

\- Noise is modeled (AWGN)

\- Demodulation for AM and DSB-SC

\- Low-pass filtering

\- SNR and error metric analysis

\- FFT to check things out in the frequency domain

\- BPSK digital modulation and demodulation

\- BER vs Eb/N0 analysis

\- And yes, an interactive dashboard built with Streamlit



The point is to show some real engineering concepts in communications, RF, digital signal processing, and simulation.



Engineering Motivation



You see communication systems everywhere — wireless networks, RF gear, satellites, radar, embedded tech, aerospace, defense hardware, all sorts of digital communication stuff.



This simulator walks you through the whole chain:



information source → transmitter → noisy channel → receiver → performance analysis



It's a hands-on way to see how the pieces fit together and what happens to your signal as it journeys from start to finish.







Here’s how I checked the simulator:



First, I made sure all the project modules loaded up without any issues. Then, I started generating baseband signals and took a look at their time-domain waveforms to see if things looked right. I ran an FFT on the message signal and saw clear peaks popping up at 500 Hz, 1000 Hz, and 2000 Hz—exactly what I expected.



Next, I tried AM and DSB-SC modulation and watched the signal energy shift right around the 10 kHz carrier, which confirmed the modulation worked. I added AWGN noise and noticed that lowering the SNR made the noise much more visible, just as it should.



After that, I demodulated both the AM and DSB-SC signals and compared the output to the original messages to make sure the recovery was accurate. I also ran through the numbers—calculating MSE, RMSE, NMSE, and measuring SNR to double-check performance.



For the BPSK part, I ran simulations and saw the bit error rate drop as Eb/N0 increased, which lined up with theory. Finally, I built a Streamlit dashboard where I could play around with parameters and instantly see how the results changed.

