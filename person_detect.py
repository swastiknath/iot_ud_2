'''
AUTHOR : SWASTIK NATH.
CAPSTONE 2: BUILDING A SMART QUEUE SYSTEM.
INTEL(R) EDGE AI FOR IOT DEVELOPERS NANODEGREE.
QUEUE SYSTEM AND INFERENCE MODULE.
'''
import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys, traceback


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.ie = IECore()
        try:
            self.model=self.ie.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.net_plugin = self.ie.load_network(network=self.model, device_name=self.device)
        
        
    def predict(self, image):
        prepro_img = self.preprocess_input(image)
        self.request_handler = self.net_plugin.start_async(request_id=0, inputs={self.input_name:prepro_img})
        if self.wait(0) == 0:
            output = self.net_plugin.requests[0].outputs[self.output_name]
            coords, image = self.draw_outputs(output, image)
        return coords, image
    
    def draw_outputs(self, coords, image):
        coords_new = []
        for obj in coords[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                color = (210, 0, 100)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
                coords_new.append((xmin, ymin, xmax, ymax))
#                 coords_new.append((int(obj[3]), int(obj[4]), int(obj[5]), int(obj[6])))
        return coords_new, image  
        

    def preprocess_outputs(self, output):
        raise NotImplementedError
            

    def preprocess_input(self, image):
        n,c,h,w = self.input_shape
        image = np.copy(image)
        inf_image = cv2.resize(image, (w, h))
        inf_image = inf_image.transpose((2,0,1))
        inf_image = inf_image.reshape((n,c,h,w))
        return inf_image
    
    def wait(self, request_id):
        ###  Implementing the Infinite Wait for Asynchronous Request to deal with the [REQUEST_BUSY] Error
        wait_for_complete_interface = self.net_plugin.requests[request_id].wait(-1)
        return wait_for_complete_interface

def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    global initial_h, initial_w
    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print( "*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print ("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)