from moviepy.editor import VideoFileClip

def remove_first_frames(video_path, output_path, frames_to_remove=29):
    """
    Remove the first 'frames_to_remove' frames from the video.

    Args:
    video_path (str): Path to the input video file.
    output_path (str): Path where the output video will be saved.
    frames_to_remove (int): Number of frames to remove from the start of the video.

    Returns:
    str: Message indicating the completion and the location of the output file.
    """

    # Load the video
    clip = VideoFileClip(video_path)

    # Calculate the duration of frames to remove
    duration_to_remove = frames_to_remove / clip.fps

    # Cut the video from the duration to remove to the end
    modified_clip = clip.subclip(duration_to_remove, clip.duration)

    # Write the modified clip to the output file
    modified_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return f"Video modified. First {frames_to_remove} frames removed. Output saved to: {output_path}"

# Example usage of the function
remove_first_frames("/Users/youkipepper/Desktop/pbMoMa/media_attached/221022_test_darkest_edge_roi(1067,1687,175,179)_ROI(1037,1578,437,339).mp4", "/Users/youkipepper/Desktop/pbMoMa/media_attached/221022_test_darkest_edge_roi(1067,1687,175,179)_ROI(1037,1578,437,339).mp4")
