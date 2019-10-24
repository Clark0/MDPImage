import os
import cv2
import grpc
import time
import hashlib
import imghdr
from concurrent import futures
from ImageDetector import ImageDetector

import message_pb2
import message_pb2_grpc

IMAGE_DIR = "img"

class ImageDetectorServicer(message_pb2_grpc.ImageDetectorServicer):
    def __init__(self):
        self.image_detector = ImageDetector()

    def DetectImage(self, request_iterator, context):
        """
        Arguments:
            request_iterator (iterator): An iterator of incoming requests.
            context: The gRPC connection context.
        """
        data_blocks = []
        data_hash = None
        name = None
        for request in request_iterator:
            m = hashlib.new('md5', request.data_block).hexdigest()
            if m != request.data_block_hash:
                return error_handler.throw_exception(
                    grpc_context=context,
                    code=grpc.StatusCode.DATA_LOSS,
                    details='DATA_LOSS: Datablock is corrupted.'
                )
            
            data_hash = request.data_hash
            name = request.name

            data_blocks.append(request.data_block)
            if len(data_blocks) > 100:
                return error_handler.throw_exception(
                    grpc_context=context,
                    code=grpc.StatusCode.FAILED_PRECONDITION,
                    details='FAILED_PRECONDITION: Image is oversized.'
                )
        
        data = b''.join(data_blocks)
        m = hashlib.new('md5', data).hexdigest()
        if m != data_hash:
            return error_handler.throw_exception(
                    grpc_context=context,
                    code=grpc.StatusCode.DATA_LOSS,
                    details='DATA_LOSS: Data is corrupted.'
                )

        photo_format = imghdr.what('', data)
        filename = name.replace('/', '')
        if photo_format:
            if not os.path.exists(IMAGE_DIR):
                os.mkdir(IMAGE_DIR)

            filename = '{}.{}'.format(filename, photo_format)
            with open(os.path.join(IMAGE_DIR, filename), 'wb') as f:
                f.write(data)

            image_np = cv2.imread(os.path.join(IMAGE_DIR, filename), cv2.IMREAD_COLOR)
            image_np, label, score = self.image_detector.detect(image_np, filename)
            cv2.imwrite(os.path.join(IMAGE_DIR, filename), image_np)
            if score > 0.6:
                return message_pb2.DetectResult(row=1, col=1, id=label)
            else:
                return message_pb2.DetectResult(row=1, col=1, id=0)

        else:
            return error_handler.throw_exception(
                    grpc_context=context,
                    code=grpc.StatusCode.FAILED_PRECONDITION,
                    details='FAILED_PRECONDITION: File type is not supported.'
                )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    message_pb2_grpc.add_ImageDetectorServicer_to_server(
        ImageDetectorServicer(), server)

    server.add_insecure_port('0.0.0.0:8008')
    server.start()
    print('API server started. Listening at 0.0.0.0:8008.')
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
