import time
import grpc
import hashlib
import message_pb2_grpc
import message_pb2

SERVER_ADDRESS = 'localhost'
PORT = 8008
BLOCK_SIZE = 20000

class PhotoDataBlockRequestIterable(object):
    def __init__(self, name, photo_path):
        self.name = name
        self.photo_path = photo_path
        
        with open(photo_path, 'rb') as f:
            self.data = f.read()
        self.data_hash = hashlib.new('md5', self.data).hexdigest()
        self.loc = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        data_block = self.data[self.loc:self.loc + BLOCK_SIZE]
        if data_block:
            data_block_hash = hashlib.new('md5', data_block).hexdigest()
            print(data_block_hash)
            request = message_pb2.ImageDataBlock(
                name = self.name,
                data_block = data_block,
                data_block_hash = data_block_hash,
                data_hash = self.data_hash
            )
            self.loc += BLOCK_SIZE
            return request
        else:
            raise StopIteration

class ImageClient(object):
    def __init__(self):
        """Initializer. 
           Creates a gRPC channel for connecting to the server.
           Adds the channel to the generated client stub.
        Arguments:
            None.
        
        Returns:
            None.
        """
        self.channel = grpc.insecure_channel(f'{SERVER_ADDRESS}:{PORT}')
        self.stub = message_pb2_grpc.ImageDetectorStub(self.channel)
        
    def upload_photo(self, name, photo_path):
        """Uploads a photo.
        Arguments:
            name: The resource name of a photo.
            photo_path: The path to a binary image file.
        
        Returns:
            None; outputs to the terminal.
        """
        def process_response(future):
            print(future.result())

        data_block_iterable = PhotoDataBlockRequestIterable(name, photo_path)

        try:
            future = self.stub.DetectImage.future(data_block_iterable)
            future.add_done_callback(process_response)
            print(1)
        except grpc.RpcError as err:
            print(err.details())
            #print('{}, {}'.format(err.code().name, err.code().value()))

if __name__ == '__main__':
    imageClient = ImageClient()
    imageClient.upload_photo("test", "1570946375.jpg")
    print(11)
    while True:
        time.sleep(100000)
