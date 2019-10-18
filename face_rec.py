from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import sys


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--cascade', default='haarcascade_frontalface_default.xml', help='path to where the face cascade resides')
    parser.add_argument('-e', '--encodings', default='encodings.pickle', help='path to serialized db of facial encodings')
    parser.add_argument('-d', '--detection-method', default='cnn', help='face detection model to use: either `haar (haarcascade)` or `cnn`')
    parser.add_argument('-i', '--video-input', default=None)
    parser.add_argument('-o', '--video-output', default="test.avi")
    return parser.parse_args(args)


def recognize_faces(data, encodings):
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    return names


def detect_faces(frame, detection_method, detector):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if detection_method == 'haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    else:
        boxes = face_recognition.face_locations(rgb, model=detection_method)

    # compute the facial embeddings for each face bounding box
    return boxes, face_recognition.face_encodings(rgb, boxes)


def mark_faces(frame, boxes, names):
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)


def start_video_stream(load_video):
    if load_video:
        print("[INFO] processing video...")
        vs = cv2.VideoCapture(load_video)

    else:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
    fps = FPS().start()
    save_video = None
    return vs, save_video, fps


def try_to_save(load_video, video_output_name, writer):
    if load_video:
        if writer is None and video_output_name is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_output_name, fourcc, 24,
                                     (frame.shape[1], frame.shape[0]), True)
        if writer is not None:
            writer.write(frame)


def main(*args):
    args = parse_args(args)
    # load the known faces and embeddings
    print("[INFO] loading encodings + face detector...")
    encodings_data = pickle.loads(open(args.encodings, "rb").read())

    # TW need for classifier ????
    # OpenCV's Haar cascade for face detection only for raspberypi
    if args.detection_method == 'haar':
        detector = cv2.CascadeClassifier('/Users/tomwit/git_home/face-recognition/haarcascade_frontalface_default.xml')
    vs, writer, fps = start_video_stream(args.video_input)

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)

        if args.video_input:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
        else:
            frame = vs.read()

        frame = imutils.resize(frame, width=500)
        boxes, encodings = detect_faces(frame=frame, detection_method=args.detection_method, detector=detector)

        names = recognize_faces(encodings_data, encodings)
        mark_faces(frame, boxes, names)

        try_to_save(load_video=args.video_input, video_output_name=args.video_output, writer=writer)

        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    if args.video_input:
        vs.release()
        # check to see if the video writer point needs to be released
        if writer is not None:
            writer.release()
    else:
        vs.stop()


if __name__ == "__main__":
    main(*(sys.argv[1:]))
