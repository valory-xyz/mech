# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains the implementation of the short_maker tool."""

import json
import logging
import math
import os
from typing import Dict, Any, Tuple, Optional, List

from moviepy.audio.fx.audio_fadeout import audio_fadeout
from openai import OpenAI

import requests
from aea_cli_ipfs.ipfs_utils import IPFSTool
from moviepy.audio.AudioClip import concatenate_audioclips, AudioClip
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from moviepy.video.VideoClip import ColorClip, ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from replicate import Client

ALLOWED_TOOLS = [
    "short-maker",
]
TOOL_TO_ENGINE = {tool: "gpt-3.5-turbo" for tool in ALLOWED_TOOLS}


def download_file(url: str, local_filename: str):
    """Utility function to download a file from a URL to a local path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def get_audio_prompts(user_input: str, engine: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """Construct the message that includes the user input"""
    message = {
        "role": "system",
        "content": f'Based on the USER INPUT: "{user_input}", please provide a short voiceover script, a prompt for generating a short soundtrack to go with the script, as well as a choice of voice for the voiceover. Format your response as a JSON object with two fields: "voiceover_script" and "soundtrack_prompt". Each field should contain the respective content as described below.\n\n'
        '- For the "voiceover_script": Use the user input to create a short script no longer than 10 seconds long (around 15-30 words), which has a mindblowing insight. The script should be in direct speech format suitable for text-to-speech AI without any stage directions. Do not just repeat the user input\n\n'
        '- For the "soundtrack_prompt": Devise a prompt that would guide an AI to generate a soundtrack that captures the mood implied by the user input.\n\n'
        "- For the \"voice\": Choose what type of voice would suit the video. If the video deals with emotional or technological themes output 'angie'. If it deals with deep questions, output 'freeman'. If it deals with humourous themes, output 'tom'. Otherwise, output 'halle'. These are the names of the voices for the model. Only output the name, nothing else. \n\n"
        "Please structure your response in the following JSON format:\n\n"
        "{\n"
        '  "voiceover_script": "[Insert narrative script here]",\n'
        '  "soundtrack_prompt": "[Insert soundtrack prompt here]",\n'
        '  "voice": "[Insert name of voice here]",\n'
        "}\n\n"
        "EXAMPLE USER INPUT:\n"
        'This is an example desired response for the user input: "Elon Musk flying to mars"\n\n'
        "{\n"
        '  "voiceover_script": "In the vast expanse of space, Elon Musk propels towards Mars, not just traversing distance, but also the boundaries of human ambition. This journey symbolizes a leap into the unknown, igniting dreams of interplanetary existence.",\n'
        '  "soundtrack_prompt": "Create an awe-inspiring and futuristic soundtrack that combines elements of space-themed ambience, such as soft electronic tones and ethereal sounds, with a hint of suspense to reflect the groundbreaking venture of flying to Mars.",\n'
        '  "voice": "freeman",\n'
        "}\n\n",
    }
    try:
        # Send the message to the chat completions endpoint
        response = client.chat.completions.create(model=engine, messages=[message])

        # Parse the JSON content from the response
        content = response.choices[0].message.content

        # Load the content as a JSON object to ensure proper JSON formatting
        json_object = json.loads(content)

        print(json_object)
        return json_object

    except Exception as e:
        logging.error("Failed to get a response from OpenAI: %s", e)
        raise


def get_shot_prompts(
    user_input: str, voiceover_length: float, engine: str = "gpt-3.5-turbo"
):
    """
    Sends a prompt to the OpenAI API and returns the response as a JSON object for video shot prompts, without verbs at the start.
    """

    # Calculate the number of shots needed
    number_of_shots = math.ceil(voiceover_length / 2.5)

    # Construct the message that includes the user input
    message = {
        "role": "system",
        "content": f'Based on the USER INPUT: "{user_input}", please provide {number_of_shots} prompts for generating shots in a video clip, starting directly with the type of shot or the subject, without using verbs at the beginning. Format your response as a JSON object with the following fields: "shot1_prompt", "shot2_prompt", "shot3_prompt", etc. Each field should contain a description of a shot based on the user input.\n\n'
        "Please structure your response in the following JSON format:\n\n"
        "{\n",
    }

    # Add placeholders for each shot
    for i in range(1, number_of_shots + 1):
        message[
            "content"
        ] += f'  "shot{i}_prompt": "[Insert prompt for shot{i} here]",\n'

    # Close the JSON structure
    message["content"] = message["content"].strip(",\n") + "\n}"

    try:
        # Send the message to the chat completions endpoint
        response = client.chat.completions.create(model=engine, messages=[message])

        # Parse the JSON content from the response
        content = response.choices[0].message.content

        # Load the content as a JSON object to ensure proper JSON formatting
        json_object = json.loads(content)

        print(json_object)
        return json_object
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_voiceover(client: Client, text: str, voice: str) -> str:
    """Narrate the provided text"""
    url = client.run(
        "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71",
        input={
            "seed": 0,
            "text": text,
            "preset": "fast",
            "voice_a": voice,
            "voice_b": "disabled",
            "voice_c": "disabled",
        },
    )
    return url


def process_soundtrack(client: Client, prompt: str, duration: int = 10) -> str:
    """Get a soundtrack for the provided prompt."""
    url = client.run(
        "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906",
        input={
            "seed": 3442726813,
            "top_k": 250,
            "top_p": 0,
            "prompt": prompt,
            "duration": duration + 1,
            "temperature": 1,
            "continuation": False,
            "model_version": "large",
            "output_format": "wav",
            "continuation_end": 9,
            "continuation_start": 7,
            "normalization_strategy": "peak",
            "classifier_free_guidance": 3,
        },
    )
    return url


def generate_images_with_dalle(
    shot_prompts: Dict, api_key: str, engine: str = "dall-e-3"
):
    """
    uses the shot prompts to generate images from dalle. These images will be used to generate videos with stability ai
    """
    generated_images = {}
    for shot_number, prompt in shot_prompts.items():
        try:
            json_params = dict(
                model=engine,
                prompt=prompt,
                size="1024x1024",
                n=1,
            )
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Stability-Client-ID": "mechs-tool",
                },
                json=json_params,
            )
            body = response.json()
            if body["data"] and len(body["data"]) > 0:
                image_url = body["data"][0]["url"]
                generated_images[shot_number] = image_url
            else:
                generated_images[shot_number] = "No image data in response"

            print(generated_images)

        except Exception as e:
            print(f"Error generating image for {shot_number}: {e}")
            generated_images[shot_number] = str(e)

    return generated_images


def process_first_shots(client: Client, shot_url: str):
    """
    Processes the first video shots using the Replicate API based on the given video prompt.
    """
    if not shot_url or not shot_url.startswith("http"):
        print(f"Invalid URL: {shot_url}")
        return None

    try:
        video_url = client.run(
            "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
            input={
                "cond_aug": 0.02,
                "decoding_t": 14,
                "input_image": shot_url,
                "video_length": "25_frames_with_svd_xt",
                "sizing_strategy": "maintain_aspect_ratio",
                "motion_bucket_id": 127,
                "frames_per_second": 10,
            },
        )
        return video_url
    except Exception as e:
        print(f"Error processing video shot: {e}")
        return None


def process_last_shot(client: Client, shot_url: str, voiceover_length: int):
    """
    Processes the first video shots using the Replicate API based on the given video prompt.
    """
    remainder = voiceover_length % 2.5
    video_url = client.run(
        "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
        input={
            "cond_aug": 0.02,
            "decoding_t": 14,
            "input_image": shot_url,
            "video_length": "14_frames_with_svd",
            "sizing_strategy": "maintain_aspect_ratio",
            "motion_bucket_id": 127,
            "frames_per_second": math.ceil(14 / remainder),
        },
    )
    return video_url


def get_video_url(
    client: Client, prompt: str, num_frames: int = 100, fps: int = 10
) -> str:
    """Get a video for the provided prompt."""
    url = client.run(
        "wcarle/text2video-zero-openjourney:2bf28cacd1f02765bd557294ec53f743b42be123675773c810bb3e0f8e3ce6f6",
        input={"prompt": prompt, "video_length": num_frames, "fps": fps},
    )
    return url


def get_audio_duration(audio_file_path: str):
    """
    Get the duration of an audio file in seconds, rounded up to the nearest second.

    Args:
    audio_file_path (str): The file path to the audio file.

    Returns:
    int: The duration of the audio file in seconds, rounded up.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"The audio file {audio_file_path} does not exist.")

    with AudioFileClip(audio_file_path) as audio_clip:
        duration_seconds = math.ceil(audio_clip.duration)

    print(f"The audio duration is {duration_seconds}")
    return duration_seconds


def compose_final_video(
    video_shots: List,
    voiceover_path: str,
    soundtrack_path: str,
    file_prefix: str,
    voiceover_volume: float = 1.0,
    soundtrack_volume: float = 0.2,
    fadeout_duration: float = 1.0,
) -> str:
    """Compose the final video."""
    clips = [VideoFileClip(shot) for shot in video_shots]
    final_video_clip = concatenate_videoclips(clips)

    # Audio Clips
    voiceover_clip = AudioFileClip(voiceover_path).volumex(voiceover_volume)
    soundtrack_clip = AudioFileClip(soundtrack_path).volumex(soundtrack_volume)

    # Determine the longest duration
    print(f"Original Video Duration: {final_video_clip.duration}")
    print(f"Original Voiceover Duration: {voiceover_clip.duration}")
    print(f"Original Soundtrack Duration: {soundtrack_clip.duration}")

    longest_duration = max(
        final_video_clip.duration, voiceover_clip.duration, soundtrack_clip.duration
    )

    # Extend video clip with a black frame if needed
    if final_video_clip.duration < longest_duration:
        black_clip = ColorClip(
            size=final_video_clip.size,
            color=(0, 0, 0),
            duration=longest_duration - final_video_clip.duration,
        )
        final_video_clip = concatenate_videoclips([final_video_clip, black_clip])
        print(f"Extended Video Duration: {final_video_clip.duration}")

    # Function to create silence
    make_silence = lambda t: [0]

    # Extend audio clips with silence if needed
    if voiceover_clip.duration < longest_duration:
        silence_duration = longest_duration - voiceover_clip.duration
        silence_clip = AudioClip(
            make_frame=make_silence, duration=silence_duration
        ).set_fps(44100)
        voiceover_clip = concatenate_audioclips([voiceover_clip, silence_clip])
        print(f"Extended Voiceover Duration: {voiceover_clip.duration}")
    if soundtrack_clip.duration < longest_duration:
        silence_duration = longest_duration - soundtrack_clip.duration
        silence_clip = AudioClip(
            make_frame=make_silence, duration=silence_duration
        ).set_fps(44100)
        soundtrack_clip = concatenate_audioclips([soundtrack_clip, silence_clip])
        print(f"Extended Soundtrack Duration: {soundtrack_clip.duration}")

    # Apply fade out to the soundtrack
    soundtrack_clip = soundtrack_clip.fx(audio_fadeout, fadeout_duration)

    final_audio = CompositeAudioClip([soundtrack_clip, voiceover_clip])

    final_video = final_video_clip.set_audio(final_audio)

    filename = f"{file_prefix}.mp4"
    final_video.write_videofile(filename, codec="libx264", audio_codec="aac", fps=24)

    return filename


def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Run the task"""
    user_input = kwargs["prompt"]
    openai_key = kwargs["api_keys"]["openai"]

    # Initialize OpenAI client with the provided key
    global client
    client = OpenAI(api_key=openai_key)

    replicate_key = kwargs["api_keys"]["replicate"]
    client_replicate = Client(replicate_key)

    file_prefix = user_input[:5]  # Extract first 5 characters

    # Step 2: Get audio prompts
    audio_prompts = get_audio_prompts(user_input)

    # Step 3: Process voiceover
    voiceover_script = audio_prompts["voiceover_script"]
    voice_choice = audio_prompts["voice"]
    voiceover = process_voiceover(client_replicate, voiceover_script, voice_choice)

    # Download voiceover and get duration
    voiceover_filename = f"{file_prefix}_voiceover.mp3"
    voiceover_path = download_file(voiceover, voiceover_filename)
    voiceover_length = get_audio_duration(voiceover_path)

    # Step 5: Get shot prompts
    shot_prompts = get_shot_prompts(user_input, voiceover_length)

    # Step 6: Generate images with DALL-E
    images = generate_images_with_dalle(shot_prompts, openai_key)
    image = list(images.values())[0]
    first_shot_path = download_file(image, "first_shot")

    # Steps 7 & 8: Process video shots
    video_urls = [process_first_shots(client_replicate, url) for url in images.values()]

    # Step 9: Process soundtrack
    soundtrack_prompt = audio_prompts["soundtrack_prompt"]
    soundtrack = process_soundtrack(
        client_replicate, soundtrack_prompt, voiceover_length
    )

    # Download all video shots and soundtrack
    video_files = [
        download_file(url, f"{file_prefix}_shot_{i}.mp4")
        for i, url in enumerate(video_urls)
    ]
    soundtrack_filename = f"{file_prefix}_soundtrack.mp3"
    soundtrack_path = download_file(soundtrack, soundtrack_filename)

    # Step 10: Compose final video
    final_video = compose_final_video(
        video_files, voiceover_path, soundtrack_path, file_prefix
    )

    ipfs_tool = IPFSTool()
    _, video_hash_, _ = ipfs_tool.add(final_video, wrap_with_directory=False)
    image_hash_ = ipfs_tool.client.add(
        first_shot_path, cid_version=1, wrap_with_directory=False
    )["Hash"]

    print(f"Stored the output on: {video_hash_}")

    body = {
        "video": video_hash_,
        "image": image_hash_,
        "prompt": user_input,
    }
    return json.dumps(body), user_input, None
