from flask import Flask, render_template, Response, request, flash, url_for, redirect
import os
from pathlib import Path
import cv2
import face_recognition
import pickle
import datetime
from cachetools import TTLCache
from .facerec.train_faces import trainer
import subprocess
import base64

cache = TTLCache(maxsize=20, ttl=10)
BASE_DIR = Path(__file__).resolve().parent
PEOPLE_FOLDER = os.path.join(BASE_DIR, "static/capture_image")
STATIC_PATH = os.path.join(BASE_DIR, "static")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = PEOPLE_FOLDER
app.config["SECRET_KEY"] = "dev"
count = 0


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


def identify1(frame, name, buf, buf_length, known_conf):
    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)

    if count >= known_conf:
        timestamp = datetime.datetime.now()
        print(name, timestamp)
        cache[name] = "detected"
        with open(os.path.join(STATIC_PATH, "log.txt"), "a") as f:
            f.write(f"{name}: {timestamp}<br>\n")
        path = "detected/{}_{}.jpg".format(name, timestamp)
        write_path = "media/" + path
        cv2.imwrite(write_path, frame)


def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path"
        )

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, "rb") as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(
        rgb_frame, number_of_times_to_upsample=2
    )

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        rgb_frame, known_face_locations=X_face_locations
    )

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # print(closest_distances)
    are_matches = [
        closest_distances[0][i][0] <= distance_threshold
        for i in range(len(X_face_locations))
    ]
    # print(are_matches)
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [
        (pred, loc) if rec else ("unknown", loc)
        for pred, loc, rec in zip(
            knn_clf.predict(faces_encodings), X_face_locations, are_matches
        )
    ]


def identify_faces(video_capture):
    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0

    process_this_frame = True
    global count
    while True:
        if count == 10:
            count = 0
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(
                rgb_frame, model_path="facerecoproj/facerec/models/trained_model.clf"
            )
        process_this_frame = not process_this_frame
        face_names = []

        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


            if name == "unknown":
                # Save Frame into disk using imwrite method
                # dir = os.listdir(os.path.join(Path(__file__).resolve().parent, "static/capture_image/"))
                # if len(dir) == 0: 
                #     count=0
                
                if count != 10:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fm = variance_of_laplacian(gray)
                    # if the focus measure is less than the supplied threshold,
                    # then the image should be considered "blurry"
                    if fm < 100:
                        continue
                    # cv2.imwrite(
                    #     os.path.join(
                    #         Path(__file__).resolve().parent,
                    #         "static/capture_image/Frame" + str(count) + ".jpg",
                    #     ),
                    #     frame,
                    # )
                    count += 1
            face_names.append(name)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            # person not identify capture the image
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )
            identify1(frame, name, buf, buf_length, known_conf)
            face_names.append(name)

        buf[i] = face_names
        i = (i + 1) % buf_length

        # print(buf)
        # Display the resulting image
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # concat frame one by one and show result

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


@app.route("/video_feed")
def video_feed():
    video_capture = cv2.VideoCapture(0)
    return Response(
        identify_faces(video_capture),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method == "POST":
        name = request.form['name']
        name = name.replace(' ', '_')
        DATASET_FOLDER = os.path.join(BASE_DIR, f"facerec/dataset/{name}")
        data_url = request.form['data']
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        if not os.path.exists(DATASET_FOLDER):
            os.makedirs(DATASET_FOLDER)
            num_files = len(os.listdir(DATASET_FOLDER))
        else:
            num_files = len(os.listdir(DATASET_FOLDER))
        save_path = os.path.join(BASE_DIR, f"facerec/dataset/{name}/{name+'_'+str(num_files)+'.jpg'}")
        with open(save_path, "wb") as fh:
            fh.write(body)

        trainer()
        flash(f"File Uploaded name and your name is %s" % name)
        return redirect(url_for("identify"))
    return render_template("upload.html")


@app.route("/")
def identify():
    return render_template("index.html")


@app.route("/train-model")
def train_model():
    trainer()
    return redirect(url_for("identify"))
