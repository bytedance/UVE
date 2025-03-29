############ Text-Video Alignment  ############

TV_ALIGNMENT_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description.

Complete your evaluation by answering this question:
Is the video content perfectly aligned with this text description: "${source}"?
Please directly answer yes or no:
"""

TV_ALIGNMENT_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description.

Complete your evaluation by answering this question:
How does the video content align with this text description: "${source}"?
Please answer good or bad:
"""

TV_ALIGNMENT_SINGLE_SOFT_SIMPLE = """
Is the video content aligned with this text description: "${source}"?
Please directly answer yes or no:
"""

TV_ALIGNMENT_MOTION_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of motion. Motion may include human actions, object movements, and camera movements.

Complete your evaluation by answering this question:
Is the video content perfectly aligned with this text description in terms of motion: "${source}"?
Please directly answer yes or no:
"""

TV_ALIGNMENT_MOTION_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of motion. Motion may include human actions, object movements, and camera movements.

Complete your evaluation by answering this question:
How does the video content align with this text description: "${source}"?
Please answer good or bad:
"""

TV_ALIGNMENT_APPEARANCE_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of subject (e.g., a person, an animal, a vehicle or an object) and scene appearance.

Complete your evaluation by answering this question:
Is the video content perfectly aligned with this text description in terms of appearance: "${source}"?
Please directly answer yes or no:
"""

TV_ALIGNMENT_APPEARANCE_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of subject (e.g., a person, an animal, a vehicle or an object) and scene appearance.

Complete your evaluation by answering this question:
How does the video content align with this text description: "${source}"?
Please answer good or bad:
"""

TV_ALIGNMENT_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the this text description: "${source}".

Please directly output a discrete number from 1 to 100, where 1 means "Completely Misaligned" and 100 means "Perfectly Aligned":
"""

TV_ALIGNMENT_APPEARANCE_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of subject (e.g., a person, an animal, a vehicle or an object) and scene appearance. 

Text description: "${source}"

Please directly output a discrete number from 1 to 100, where 1 means "Completely Misaligned" and 100 means "Perfectly Aligned":
"""

TV_ALIGNMENT_MOTION_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate how well the video content aligns with the following text description in terms of motion. Motion may include human actions, object movements, and camera movements. 

Text description: "${source}"

Please directly output a discrete number from 1 to 100, where 1 means "Completely Misaligned" and 100 means "Perfectly Aligned":
"""

TV_ALIGNMENT_PAIR = """
Watch the above two AI-generated videos and evaluate how well their video content aligns with the following text description.

Complete your evaluation by answering this question:
Which video is more aligned with "${source}"?

you should make your judgement based on the following rules:
    - If the first video aligns better with the text, answer "the first video" 
    - Else, if the second video aligns better with the text, answer "the second video". 
    - Else, if both videos align equally well with the text, answer "same good". 
    - Else, if neither video aligns well with the text, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same good", or "same bad".

Now give your judgement:
"""

TV_ALIGNMENT_PAIR_SIMPLE = """
Which video is more aligned with "${source}"?
Answer "the first video", "the second video", "same good", or "same bad":
"""

TV_ALIGNMENT_APPEARANCE_PAIR = """
Watch the above two AI-generated videos and evaluate how well their video content aligns with the following text description.

Focus on video-text alignment in terms of subject (e.g., a person, an animal, a vehicle or an object) and scene appearance.

Complete your evaluation by answering this question:
Which video is more aligned with "${source}"?  

you should make your judgement based on the following rules:
    - If the first video aligns better with the text, answer "the first video" 
    - Else, if the second video aligns better with the text, answer "the second video". 
    - Else, if both videos align equally well with the text, answer "same good". 
    - Else, if neither video aligns well with the text, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same good", or "same bad".

Now give your judgement:
"""

TV_ALIGNMENT_MOTION_PAIR = """
Watch the above two AI-generated videos and evaluate how well their video content aligns with the following text description.

Focus on video-text alignment in terms of motion (e.g., human actions, object movements, and camera movements).

Complete your evaluation by answering this question:
Which video is more aligned with "${source}"? 

you should make your judgement based on the following rules:
    - If the first video aligns better with the text, answer "the first video" 
    - Else, if the second video aligns better with the text, answer "the second video". 
    - Else, if both videos align equally well with the text, answer "same good". 
    - Else, if neither video aligns well with the text, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same good", or "same bad".

Now give your judgement:
"""

############ Dynamic Degree  ############

DYNAMIC_DEGREE_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Pay attention to three aspects: camera movement, subject (e.g., a person, an animal, a vehicle or an object) movement and changes in color and lighting conditions. If any of these aspects are highly dynamic, consider the dynamic degree of the video high.

Complete your evaluation by answering this question:
Is the dynamic degree of the video content high? 
Please directly answer yes or no:
"""

DYNAMIC_DEGREE_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Pay attention to three aspects: camera movement, subject (e.g., a person, an animal, a vehicle or an object) movement and changes in color and lighting conditions. If any of these aspects are highly dynamic, consider the dynamic degree of the video high.

Complete your evaluation by answering this question:
How is the dynamic degree of the video content?
Please answer good or bad:
"""

DYNAMIC_DEGREE_SINGLE_SOFT_SIMPLE = """"
Is the dynamic degree of the video content high? 
Please directly answer yes or no:
"""

DYNAMIC_DEGREE_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Pay attention to three aspects: camera movement, subject (e.g., a person, an animal, a vehicle or an object) movement and changes in color and lighting conditions. If any of these aspects are highly dynamic, consider the dynamic degree of the video high.

Please directly output a discrete number from 1 to 100, where 1 means "Very Low" dynamic degree and 100 means "Very High" dynamic degree:
"""

LIGHT_CHANGE_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the change of lighting conditions and colors.

Complete your evaluation by answering this question:
Is the video displaying significant changes in lighting conditions or colors?
Please directly answer yes or no:
"""

LIGHT_CHANGE_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the change of lighting conditions and colors.

Complete your evaluation by answering this question:
How is the degree of light and color change in the video?
Please answer good or bad:
"""

LIGHT_CHANGE_SINGLE_SOFT_SIMPLE = """
Is the light and color change of the video large?
Please directly answer yes or no:
"""

LIGHT_CHANGE_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the change of lighting conditions and colors.

Please directly output a discrete number from 1 to 100, where 1 means "Very Small" change of lighting conditions and colors and 100 means "Very Large" change of lighting conditions and colors:
"""

CAMERA_MOTION_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on camera movement.

Complete your evaluation by answering this question:
Is the camera movement of the video displaying high motion dynamics?
Please directly answer yes or no:
"""

CAMERA_MOTION_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on camera movement.

Complete your evaluation by answering this question:
How is the degree of camera movement in the video?
Please answer good or bad:
"""

CAMERA_MOTION_SINGLE_SOFT_SIMPLE = """
Is the camera movement of the video large?
Please directly answer yes or no:
"""

CAMERA_MOTION_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on camera movement.

Please directly output a discrete number from 1 to 100, where 1 means "Very Small" camera movement and 100 means "Very Large" camera movement:
"""

SUBJECT_MOTION_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the movement of subjects (e.g., a person, an animal, a vehicle or an object).

Complete your evaluation by answering this question:
Is the subject movement in the video displaying high motion dynamics? 
Please directly answer yes or no:
"""

SUBJECT_MOTION_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the movement of subjects (e.g., a person, an animal, a vehicle or an object).

Complete your evaluation by answering this question:
How is the degree of subject movement in the video?
Please answer good or bad:
"""

SUBJECT_MOTION_SINGLE_SOFT_SIMPLE = """
Is the subject movement of the video large? 
Please directly answer yes or no:
"""

SUBJECT_MOTION_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the dynamic degree of the video.

Please focus on the movement of subjects (e.g., a person, an animal, a vehicle or an object).

Please directly output a discrete number from 1 to 100, where 1 means "Very Small" subject movement and 100 means "Very Large" subject movement:
"""

DYNAMIC_DEGREE_PAIRWISE = """
Watch the above two AI-generated videos and evaluate their dynamic degree.

Pay attention to three aspects: camera movement, subject (e.g., a person, an animal, a vehicle or an object) movement and changes in color and lighting conditions. If any of these aspects are highly dynamic, consider the dynamic degree of the video high.

Complete your evaluation by answering this question:
Which video displays higher dynamic degree?

you should make your judgement based on the following rules:
    - If the first video is more dynamic, answer "the first video"
    - Else, if the second video is more dynamic, answer "the second video". 
    - Else, if both videos display equally high dynamic, answer "same high". 
    - Else, if both videos display equally low dynamic, answer "same low".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same low".

Now give your judgement:
"""

DYNAMIC_DEGREE_PAIRWISE_SIMPLE = """
Which video displays higher dynamic degree?
Answer "the first video", "the second video", "same good", or "same bad":
"""

LIGHT_CHANGE_PAIRWISE = """
Watch the above two AI-generated videos and evaluate their dynamic degree.

Please focus on the change of lighting conditions and colors.

Complete your evaluation by answering this question:
Which video displays larger changes in lighting conditions or colors?

you should make your judgement based on the following rules:
    - If one video displays larger light or color changes, answer "the first video" or "the second video". 
    - Else, if both videos display equally large light or color changes, answer "same high". 
    - Else, if both videos display equally small light or color changes, answer "same low".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same low".

Now give your judgement:
"""

LIGHT_CHANGE_PAIRWISE_SIMPLE = """"
Which video displays larger changes in lighting conditions or colors?
Answer "the first video", "the second video", "same good", or "same bad":
"""

CAMERA_MOTION_PAIRWISE = """
Watch the above two AI-generated videos and evaluate their dynamic degree.

Please focus on camera movement.

Complete your evaluation by answering this question:
Which video displays higher degree of camera movement?

you should make your judgement based on the following rules:
    - If one video displays higher degree of camera movement, answer "the first video" or "the second video". 
    - Else, if both videos display equally high degree of camera movement, answer "same high". 
    - Else, if both videos display equally low degree of camera movement, answer "same low".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same low".

Now give your judgement:
"""

CAMERA_MOTION_PAIRWISE_SIMPLE = """
Which video displays higher degree of camera movement?
Answer "the first video", "the second video", "same good", or "same bad":
"""

SUBJECT_MOTION_PAIRWISE = """
Watch the above two AI-generated videos and evaluate their dynamic degree.

Please focus on the movement of subjects (e.g., a person, an animal, a vehicle or an object).

Complete your evaluation by answering this question:
Which video displays higher degree of subject movement?

you should make your judgement based on the following rules:
    - If one video displays higher degree of subject movement, answer "the first video" or "the second video". 
    - Else, if both videos display equally high degree of subject movement, answer "same high".
    - Else, if both videos display equally low degree of subject movement, answer "same low".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same low".

Now give your judgement:
"""

SUBJECT_MOTION_PAIRWISE_SIMPLE = """
Which video displays higher degree of subject movement?
Answer "the first video", "the second video", "same good", or "same bad":
"""

############ Static Visual Quality  ############

STATIC_VISUAL_QUALITY_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the visual quality of each individual frame.

Pay attention to the following aspects in your assessment:
    - Technical Quality: Check for any unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.
    - Structural Correctness: Check for any abnormal subject (e.g., a person, an animal, a vehicle or an object) structures that contradict common sense, such as a human hand with three fingers or a fish with three eyes.
    - Aesthetic Quality: Evaluate if the video frames have a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
Is the visual quality of each individual frame good?
Please directly answer yes or no:
"""

STATIC_VISUAL_QUALITY_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the visual quality of each individual frame.

Pay attention to the following aspects in your assessment:
    - Technical Quality: Check for any unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.
    - Structural Correctness: Check for any abnormal subject (e.g., a person, an animal, a vehicle or an object) structures that contradict common sense, such as a human hand with three fingers or a fish with three eyes.
    - Aesthetic Quality: Evaluate if the video frames have a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
How is the the visual quality of each individual frame?
Please answer good or bad:
"""

STATIC_VISUAL_QUALITY_SINGLE_SOFT_SIMPLE = """
Is the static visual quality of the video good?
Please directly answer yes or no:
"""

STRUCTURAL_CORRECTNESS_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the structural correctness of the subjects (e.g., a person, an animal, a vehicle or an object) in each individual frame.

The subject structure is considered correct if it aligns with common sense and reality, with no abnormalities such as a human hand with three fingers or a fish with three eyes.

Complete your evaluation by answering this question:
Is the structure of the subjects in the video correct?
Please directly answer yes or no:
"""

STRUCTURAL_CORRECTNESS_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the structural correctness of the subjects (e.g., a person, an animal, a vehicle or an object) in each individual frame.

The subject structure is considered correct if it aligns with common sense and reality, with no abnormalities such as a human hand with three fingers or a fish with three eyes.

Complete your evaluation by answering this question:
How is the subject structure correctness of the video?
Please answer good or bad:
"""

STRUCTURAL_CORRECTNESS_SINGLE_SOFT_SIMPLE = """
Is the subject structural correctness of the video good?
Please directly answer yes or no:
"""

TECHNICAL_QUALITY_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the technical visual quality of each individual frame.

The technical quality is considered good if the frame is free from unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.

Complete your evaluation by answering this question:
Is the technical quality of the video good?
Please directly answer yes or no:
"""

TECHNICAL_QUALITY_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the technical visual quality of each individual frame.

The technical quality is considered good if the frame is free from unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.

Complete your evaluation by answering this question:
How is the technical quality of the video?
Please answer good or bad:
"""

TECHNICAL_QUALITY_SINGLE_SOFT_SIMPLE = """
Is the technical quality of the video good?
Please directly answer yes or no:
"""

AESTHETIC_QUALITY_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the aesthetic quality of each individual frame.

The aesthetic quality is considered good if the frame has a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
Is the aesthetic quality of the video good?
Please directly answer yes or no:
"""

AESTHETIC_QUALITY_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the aesthetic quality of each individual frame.

The aesthetic quality is considered good if the frame has a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
How is the aesthetic quality of the video?
Please answer good or bad:
"""

AESTHETIC_QUALITY_SINGLE_SOFT_SIMPLE = """
Is the aesthetic quality of the video good?
Please directly answer yes or no:
"""

STATIC_VISUAL_QUALITY_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the visual quality of each individual frame.

Pay attention to the following aspects in your assessment:
    - Technical Quality: Check for any unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.
    - Structural Correctness: Check for any abnormal subject (e.g., a person, an animal, a vehicle or an object) structures that contradict common sense, such as a human hand with three fingers or a fish with three eyes.
    - Aesthetic Quality: Evaluate if the video frames have a well-structured layout, rich and harmonious colors, and is visually appealing.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" visual quality and 100 means "Perfect" visual quality:
"""

AESTHETIC_QUALITY_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the aesthetic quality of each individual frame.

The aesthetic quality is considered good if the frame has a well-structured layout, rich and harmonious colors, and is visually appealing.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" aesthetic quality and 100 means "Perfect" aesthetic quality:
"""

TECHNICAL_QUALITY_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the technical visual quality of each individual frame.

The technical quality is considered good if the frame is free from unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" technical quality and 100 means "Perfect" technical quality:
"""

STRUCTURAL_CORRECTNESS_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the structural correctness of the subjects in each individual frame.

The subject structure is considered correct if it aligns with common sense and reality, with no abnormalities such as a human hand with three fingers or a fish with three eyes.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" structural correctness and 100 means "Perfect" structural correctness:
"""

STATIC_VISUAL_QUALITY_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of the visual quality of individual frames.

Pay attention to the following aspects in your assessment:
    - Technical Quality: Check for any unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.
    - Structural Correctness: Check for any abnormal subject (e.g., a person, an animal, a vehicle or an object) structures that contradict common sense, such as a human hand with three fingers or a fish with three eyes.
    - Aesthetic Quality: Evaluate if the video frames have a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
Which video has better visual quality?

you should make your judgement based on the following rules:
    - If the visual quality of the first video is better than the second video, answer "the first video".
    - Else, if the visual quality of the second video is better than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good visual quality, answer "same good".
    - Else, if both videos demonstrate equally bad visual quality, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

STATIC_VISUAL_QUALITY_PAIRWISE_SIMPLE = """
Which video has better static visual quality?
Answer "the first video", "the second video", "same good", or "same bad":
"""

AESTHETIC_QUALITY_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of the aesthetic quality of individual frames.

The aesthetic quality is considered good if the frame has a well-structured layout, rich and harmonious colors, and is visually appealing.

Complete your evaluation by answering this question:
Which video has better aesthetic quality?

you should make your judgement based on the following rules:
    - If the aesthetic quality of the first video is better than the second video, answer "the first video".
    - Else, if the aesthetic quality of the second video is better than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good aesthetic quality, answer "same good".
    - Else, if both videos demonstrate equally bad aesthetic quality, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

AESTHETIC_QUALITY_PAIRWISE_SIMPLE = """
Which video has better aesthetic quality?
Answer "the first video", "the second video", "same good", or "same bad":
"""

TECHNICAL_QUALITY_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of the technical quality of individual frames.

The technical quality is considered good if the frame is free from unwanted noise, blur and distortion that negatively affect visual quality, excluding those deliberately created ones such as blurring visual effects.

Complete your evaluation by answering this question:
Which video has better technical quality?

you should make your judgement based on the following rules:
    - If the technical quality of the first video is better than the second video, answer "the first video".
    - Else, if the technical quality of the second video is better than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good technical quality, answer "same good".
    - Else, if both videos demonstrate equally bad technical quality, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

TECHNICAL_QUALITY_PAIRWISE_SIMPLE = """
Which video has better technical quality?
Answer "the first video", "the second video", "same good", or "same bad":
"""

STRUCTURAL_CORRECTNESS_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of the structural correctness of subjects (e.g., a person, an animal, a vehicle or an object) in each individual frame.

The subject structure is considered correct if it aligns with common sense and reality, with no abnormalities such as a human hand with three fingers or a fish with three eyes.

Complete your evaluation by answering this question:
Which video has better structual correctness?

you should make your judgement based on the following rules:
    - If the structual correctness of the first video is better than the second video, answer "the first video".
    - Else, if the structual correctness of the second video is better than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good structual correctness, answer "same good".
    - Else, if both videos demonstrate equally bad structual correctness, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

STRUCTURAL_CORRECTNESS_PAIRWISE_SIMPLE = """
Which video has better subject structual correctness?
Answer "the first video", "the second video", "same good", or "same bad":
"""

############ Temporal Visual Quality  ############

TEMPORAL_VISUAL_QUALITY_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate the visual quality from the temporal perspective.

Pay attention to the following aspects in your assessment:
    - Appearance Consistency: Is the video free of unwanted changes in the appearance of subjects (e.g., people, animals, vehicles or objects) or scenes across frames, such as the change in a person's identity or an abrupt disappearance of objects in the background.
    - Temporal Flickering and Jittering: Is the video free of unwanted temporal flickering and jitterring that negatively affect visual quality.
    - Motion Naturalness: Do the subject motions and subject interactions appear natural and adhere to physical laws?

Complete your evaluation by answering this question:
Is the visual quality of the video good from the temporal perspective?
Please directly answer yes or no:
"""

TEMPORAL_VISUAL_QUALITY_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate the visual quality from the temporal perspective.

Pay attention to the following aspects in your assessment:
    - Appearance Consistency: Is the video free of unwanted changes in the appearance of subjects (e.g., people, animals, vehicles or objects) or scenes across frames, such as the change in a person's identity or an abrupt disappearance of objects in the background.
    - Temporal Flickering and Jittering: Is the video free of unwanted temporal flickering and jitterring that negatively affect visual quality.
    - Motion Naturalness: Do the subject motions and subject interactions appear natural and adhere to physical laws?

Complete your evaluation by answering this question:
How is the visual quality of the video from the temporal perspective?
Please answer good or bad:
"""

TEMPORAL_VISUAL_QUALITY_SINGLE_SOFT_SIMPLE = """
Is the temporal visual quality of the video good?
Please directly answer yes or no:
"""

TEMPORAL_VISUAL_QUALITY_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate the visual quality from the temporal perspective.

Pay attention to the following aspects in your assessment:
    - Appearance Consistency: Is the video free of unwanted changes in the appearance of subjects (e.g., people, animals, vehicles or objects) or scenes across frames, such as the change in a person's identity or an abrupt disappearance of objects in the background.
    - Temporal Flickering and Jittering: Is the video free of unwanted temporal flickering and jitterring that negatively affect visual quality.
    - Motion Naturalness: Do the subject motions and subject interactions appear natural and adhere to physical laws?

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" temporal visual quality and 100 means "Perfect" temporal visual quality:
"""

MOTION_NATURALNESS_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate it in terms of motion naturalness.

Unnatural motion may include subjects (e.g., people, animals, vehicles or objects) moving in unexpected directions or speeds, and interactions between subjects that do not adhere to physical laws.

Complete your evaluation by answering this question:
Do the subject motion of the video appear natural and adhere to physical laws?
Please directly answer yes or no:
"""

MOTION_NATURALNESS_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate it in terms of motion naturalness.

Unnatural motion may include subjects (e.g., people, animals, vehicles or objects) moving in unexpected directions or speeds, and interactions between subjects that do not adhere to physical laws.

Complete your evaluation by answering this question:
How is the subject motion naturalness the video?
Please answer good or bad:
"""

MOTION_NATURALNESS_SINGLE_SOFT_SIMPLE = """
Is the motion naturalness of the video good?
Please directly answer yes or no:
"""

MOTION_NATURALNESS_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate it in terms of motion naturalness.

Unnatural motion may include subjects (e.g., people, animals, vehicles or objects) moving in unexpected directions or speeds, and interactions between subjects that do not adhere to physical laws.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" motion naturalness and 100 means "Perfect" motion naturalness:
"""

FLICKERING_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate it in terms of temporal flickering and jitterring.

Temporal flickering and jitterring refers to the imperfect temporal consistency at local and high-frequency details.

Complete your evaluation by answering this question:
Is the video free of unwanted temporal flickering and jitterring that negatively affect visual quality?
Please directly answer yes or no:
"""

FLICKERING_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate it in terms of temporal flickering and jitterring.

Temporal flickering and jitterring refers to the imperfect temporal consistency at local and high-frequency details.

Complete your evaluation by answering this question:
How is the video quality in terms of temporal flickering and jitterring?
Please answer good or bad:
"""

FLICKERING_SINGLE_SOFT_SIMPLE = """
Is the temporal visual quality of the video good in terms of flickering?
Please directly answer yes or no:
"""

FLICKERING_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate it in terms of temporal flickering and jitterring.

Temporal flickering and jitterring refers to the imperfect temporal consistency at local and high-frequency details.

Please directly output a discrete number from 1 to 100, where 1 means "Severe" temporal flickering and jitterring and 100 means "Free from" temporal flickering and jitterrin:
"""

APPEARANCE_CONSISTENCY_SINGLE_SOFT_YN = """
Watch the above frames of an AI-generated video and evaluate it in terms of appearance consistency across frames.

Inconsistent appearance may include changes in subject (e.g., a person, an animal, a vehicle or an object) identity, body part deformation or abrupt disappearance of objects in the background. Note that appearance changes caused by natural subject movement or scene shift does not count as appearance inconsistency.

Complete your evaluation by answering this question:
Are the appearance of subjects and background consistent across frames?
Please directly answer yes or no:
"""

APPEARANCE_CONSISTENCY_SINGLE_SOFT_GOOD_BAD = """
Watch the above frames of an AI-generated video and evaluate it in terms of appearance consistency across frames.

Inconsistent appearance may include changes in subject (e.g., a person, an animal, a vehicle or an object) identity, body part deformation or abrupt disappearance of objects in the background. Note that appearance changes caused by natural subject movement or scene shift does not count as appearance inconsistency.

Complete your evaluation by answering this question:
How is the subject and background appearance consistency of the video?
Please answer good or bad:
"""

APPEARANCE_CONSISTENCY_SINGLE_SOFT_SIMPLE = """
Is the appearance consistency of the video good?
Please directly answer yes or no:
"""

APPEARANCE_CONSISTENCY_SINGLE_HARD = """
Watch the above frames of an AI-generated video and evaluate it in terms of appearance consistency across frames.

Inconsistent appearance may include changes in subject (e.g., a person, an animal, a vehicle or an object) identity, body part deformation or abrupt disappearance of objects in the background. Note that appearance changes caused by natural subject movement or scene shift does not count as appearance inconsistency.

Please directly output a discrete number from 1 to 100, where 1 means "Very Poor" appearance consistency and 100 means "Perfect" appearance consistency:
"""

TEMPORAL_VISUAL_QUALITY_PAIRWISE = """
Watch the above two AI-generated videos and evaluate their visual quality from the temporal perspective.

Pay attention to the following aspects in your assessment:
    - Appearance Consistency: Is the video free of unwanted changes in the appearance of subjects (e.g., people, animals, vehicles or objects) or scenes across frames, such as the change in a person's identity or an abrupt disappearance of objects in the background.
    - Temporal Flickering and Jittering: Is the video free of unwanted temporal flickering and jitterring that negatively affect visual quality.
    - Motion Naturalness: Do the subject motions and subject interactions appear natural and adhere to physical laws?

Complete your evaluation by answering this question:
Which video has better visual quality from the temporal perspective?

you should make your judgement based on the following rules:
    - If the temporal visual quality of the first video is better than the second video, answer "the first video".
    - Else, if the temporal visual quality of the second video is better than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good temporal visual quality, answer "same good".
    - Else, if both videos demonstrate equally bad temporal visual quality, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

TEMPORAL_VISUAL_QUALITY_PAIRWISE_SIMPLE = """
Which video has better temporal visual quality?
Answer "the first video", "the second video", "same good", or "same bad":
"""

MOTION_NATURALNESS_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of motion naturalness.

Unnatural motion may include subjects (e.g., people, animals, vehicles or objects) moving in unexpected directions or speeds, and interactions between subjects that do not adhere to physical laws.

Complete your evaluation by answering this question:
Which video demonstrates more natural subject motions that better adhere to physical laws?

you should make your judgement based on the following rules:
    - If the subject motion of the first video is more natural than the second video, answer "the first video".
    - Else, if the subject motion of the second video is more natural than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good subject motion naturalness, answer "same good".
    - Else, if both videos demonstrate equally bad subject motion naturalness, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

MOTION_NATURALNESS_PAIRWISE_SIMPLE = """
Which video demonstrates better motion naturalness?
Answer "the first video", "the second video", "same good", or "same bad":
"""

APPEARANCE_CONSISTENCY_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of appearance consistency across frames.

Inconsistent appearance may include changes in subject (e.g., a person, an animal, a vehicle or an object) identity, body part deformation or abrupt disappearance of objects in the background. Note that appearance changes caused by natural subject movement or scene shift does not count as appearance inconsistency.

Complete your evaluation by answering this question:
Which video demonstrates more consistent subject and scene appearance across frames?

you should make your judgement based on the following rules:
    - If the subject and scene appearance of the first video is more consistent than the second video, answer "the first video".
    - Else, if the subject and scene appearance of the second video is more consistent than the first video, answer "the second video".
    - Else, if both videos demonstrate equally good appearance consistency, answer "same good".
    - Else, if both videos demonstrate equally bad appearance consistency, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

APPEARANCE_CONSISTENCY_PAIRWISE_SIMPLE = """
Which video demonstrates better appearance consistency?
Answer "the first video", "the second video", "same good", or "same bad":
"""

FLICKERING_PAIRWISE = """
Watch the above two AI-generated videos and evaluate them in terms of temporal flickering and jitterring.

Temporal flickering and jitterring refers to the imperfect temporal consistency at local and high-frequency details.

Complete your evaluation by answering this question:
Which video exhibits fewer unwanted temporal flickering and jitterring that negatively affect visual quality?

you should make your judgement based on the following rules:
    - If the first video has fewer temporal flickering and jitterring than the second video, answer "the first video".
    - Else, if the second video has fewer temporal flickering and jitterring than the first video, answer "the second video".
    - Else, if both videos are free of unwanted temporal flickering and jitterring, answer "same good".
    - Else, if both videos exhibit unwanted temporal flickering and jitterring, answer "same bad".
    - Provide your answer directly as "the first video", "the second video", "same high" or "same bad".

Now give your judgement:
"""

FLICKERING_PAIRWISE_SIMPLE = """
Which video demonstrates less temporal flickering?
Answer "the first video", "the second video", "same good", or "same bad":
"""
