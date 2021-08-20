from multiprocessing import Pool, Queue
import time
import cv2

# intialize global variables for the pool processes:
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b


def detect_object(frame):
    time.sleep(1)
    detection_buffer.put(frame)


def show():
    while True:
        print("show")
        frame = detection_buffer.get()
        if frame is not None:
            cv2.imshow("Video", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


# required for Windows:
if __name__ == "__main__":

    detection_buffer = Queue()
    # 6 workers: 1 for the show task and 5 to process frames:
    pool = Pool(6, initializer=init_pool, initargs=(detection_buffer,))
    # run the "show" task:
    show_future = pool.apply_async(show)

    cap = cv2.VideoCapture(0)

    futures = []
    while True:
        ret, frame = cap.read()
        if ret:
            f = pool.apply_async(detect_object, args=(frame,))
            futures.append(f)
            time.sleep(0.025)
        else:
            break
    # wait for all the frame-putting tasks to complete:
    for f in futures:
        f.get()
    # signal the "show" task to end by placing None in the queue
    detection_buffer.put(None)
    show_future.get()
    print("program ended")
