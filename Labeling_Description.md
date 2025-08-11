**Video Annotation Protocol**

This document outlines the standard operating procedure for the manual annotation of video data to create ground-truth labels for Activities of Daily Living (ADLs).

**1. Annotation Software**
The primary tool used for this task is the VIA (VGG Image Annotator) software. VIA is a lightweight, standalone manual annotation tool that runs directly in a web browser and requires no installation or setup.

Citation: Abhishek Dutta and Andrew Zisserman. 2019. The VIA Annotation Software for Images, Audio and Video. In Proceedings of the 27th ACM International Conference on Multimedia (MM ’19), October 21–25, 2019, Nice, France. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3343031.3350535.

**2. Labeling Workflow**
The annotation process follows a straightforward, step-by-step workflow for each subject's video file.

Step 1: Launch VIA and Load Video
Open the local via.html file in a web browser (e.g., Chrome, Firefox).

From the VIA interface, navigate to Project -> Add local files to load the subject's 48-hour video file.

Step 2: Define and Annotate Activities
Navigate the Timeline: Use the main video timeline slider at the bottom of the interface to scrub through the video and identify the start and end points of different activities.

Create Labels: On the left-hand panel, define the activity labels (e.g., Self Propulsion, Transfer, Eating). VIA assigns a different color to each label's timeline channel for easy visual distinction.

Apply Labels: To label a segment, click on the desired label channel at the point where the activity begins. Drag the colored annotation box along the timeline until the activity's end point.

Refine Boundaries: Adjust the start and end boundaries of the annotation boxes by dragging their edges for sample-accurate labeling.

Use Shortcuts: Utilize the keyboard shortcuts described in the VIA documentation to streamline the process (e.g., playing/pausing the video, jumping between segments, creating/deleting segments).

**Step 3: Export Annotations**
Once the entire video has been annotated and reviewed for accuracy:

Navigate to Annotation -> Export Annotations (as json).

Save the resulting .json file. This file contains all the temporal metadata and corresponding labels for the video, which can then be used for data processing and model training.

This process is repeated for each subject's video file to generate the complete ground-truth dataset.

