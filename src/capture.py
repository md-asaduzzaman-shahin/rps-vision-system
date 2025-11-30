import cv2
import os
import time

def capture_images(class_name, num_images, output_dir):
    """
    Opens the webcam and captures images for a specific class (rock, paper, or scissors).
    """
    # Open default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create specific folder: data/raw/rock, etc.
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    print(f"\n=== Capturing {class_name} ===")
    print("Press 's' to Save an image.")
    print("Press 'q' to Quit this class.")
    
    count = 0
    # Check existing files to avoid overwriting
    existing_files = len(os.listdir(class_dir))
    count = existing_files

    while count < (existing_files + num_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the live feed
        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        
        cv2.imshow(f"Capturing {class_name} ({count} existing)", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # Save on 's'
            # Use timestamp to ensure unique filenames
            timestamp = int(time.time() * 1000)
            filename = os.path.join(class_dir, f"{class_name}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            count += 1
            
        elif key == ord('q'): # Quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {class_name}.\n")

if __name__ == "__main__":
    classes = ['rock', 'paper', 'scissors']
    
    # Default to saving in data/raw
    # Assumes script is run from project root
    base_dir = os.path.join("data", "webcam") 
    
    print(f"Images will be saved to: {base_dir}")
    
    for cls in classes:
        user_input = input(f"Do you want to capture images for '{cls}'? (y/n): ").lower()
        if user_input == 'y':
            try:
                num = int(input(f"How many images to capture for {cls}? "))
                capture_images(cls, num, base_dir)
            except ValueError:
                print("Invalid number. Skipping.")
