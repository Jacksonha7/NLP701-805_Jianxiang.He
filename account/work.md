# The work completed during this course
All the codes mentioned in this note have been uploaded to the course's GitHub repository.

## Before midterm Presentation
First, after understanding the course requirements, the first thing that came to my mind was my existing job, VSI. At that time, the purpose of my work was to reasonably utilize the textual information in visual search. 

After completing the simplest replication (not a full replication, just confirming that the conda environment was installed and the demo ran successfully), I realized that the two issues present in my previous paper could serve as the theme for this course project. One was that my algorithm did not perform ablation on the encoder, so it was a worthy question to consider what kind of encoder was the most reasonable to use. The other was why my algorithm still couldn't achieve 100% search accuracy in text-related problems. Was it a problem with the encoder or did it fail to model the problem well? Therefore, I planned to work based on this idea.

Since the implementation in my previous code repository was integrated, meaning that in a search algorithm, directly constructing objects, conducting searches for objects, and integrating the final question answering into one was not suitable for experiments. Therefore, the first thing I did was to split the algorithm.

I divided the code into three parts: object_grounding, frame_searching, and QA. Our main focus is on the frame_searching part, which is about how to make it achieve better frame searching performance. Therefore, there was no part related to QA in the replication process.

- object_grounding: I locally deployed the qwen3vl8b model for extracting objects from questions. The code is qwen_grounding.py. The output result is qwen3vl8b.json.
- keyframe_searching: When reproducing, I did not make extensive modifications to this file. I only made adjustments to adapt to the server side. The output result is R1_Tstar_PTonly_LongVideoBench_gpt4_cot_vid1.json and R1_VSI_PTonly_LongVideoBench_gpt4_cot_vid1.json

After the reproduction is completed, the obtained results are saved in the "matching_results" folder. I have updated the data in the interim report and clearly marked it. What needs to be clarified here is that the reason I placed the 'keyframe_matching.py' file in the code was not to prove that I had made modifications based on the original version, but to enable my replication results to also be reviewed by you. However, I suddenly realized that in fact, running that file still requires the dataset to be downloaded first before it can be executed. To facilitate your review, I directly ran the code on my own server and placed the results directly in the 'matching_results' folder, which makes it more convenient for you to compare the data.

At the same time, I was concurrently working on the coding for the new features.

## Up to Now
Since the reproducibility in the early stage did not retain specific log information, to eliminate any doubts, I conducted a complete new reproduction and saved the complete experimental commands, configurations, logs, outputs and results, which were all stored in the "reproduced/new" folder.

I have completed the code construction for the ablation of the encoder and conducted the experiments. The code for the experiments is placed in the "nlp805" folder, and the detailed experimental records are located in "nlp805/experiment". Meanwhile, the specific details of the experiment, the data and the conclusions have been updated in the latest interim report. 

