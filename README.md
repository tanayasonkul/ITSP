# ITSP
A summer project at IITB : Extracting musical notes from the audio file of songs


Today, there are many applications to aid learning. One famous thing to learn is a musical instrument. We thought about applying the same to the music learning process. One major issue we saw is the availability of music notes to practice songs of one’s choice. There we thought we could step in, and came up with this project.

Our project focuses on a single note instrument, violin. The idea is to take input in the form of audio files where someone has played the song, and provide music notes for that particular song.
 
The audio in wav format is converted into spectrograms. FFT is run on each small piece of audio signal(known as STFT) to obtain frequency-time data.We calculated high energy points in each bin of the spectrogram, corresponding frequency and time were given as input to DBSCAN clustering algorithm to obtain the final note frequencies. The frequencies are mapped to respective notes played on violin which are provided as pdf.

This is available to the user through a website. This was built with CSS and bootstrap for front end development, and flask for backend development.

[Main Documentation](https://docs.google.com/document/d/1mVRrFGD7AAq-SzGGa3nY2MF6OLNVps3jcX7BxqoTVr4/edit?usp=sharing)

[Presentation](https://docs.google.com/presentation/d/1ingeiEv7Jr_qB6cBkS9PF0LJq4VP1yoo9dhK5NRskG4/edit?usp=sharing)
