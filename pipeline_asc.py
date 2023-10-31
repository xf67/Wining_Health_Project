import argparse
import json
import os
import os.path as osp
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import torch
import wfdb
from tqdm import tqdm
from scipy.signal import find_peaks
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, Dataset
import acl
import time


DEVICE_ID=0
BATCH_SIZE=0#
CHANNEL=0#
HEIGHT=0#
WIDTH=0#
# error code
ACL_SUCCESS = 0
# rule for memory
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2
# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
buffer_method = {
    "in": acl.mdl.get_input_size_by_index,#根据aclmdlDesc类型的数据，获取指定输入的大小，单位为Byte。
    "out": acl.mdl.get_output_size_by_index#根据aclmdlDesc类型的数据，获取指定输出的大小，单位为Byte。
    }
def check_ret(message, ret):
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}".format(message, ret))

class Net(object):
    def __init__(self, deviceID, modelPath,batch_size=1):
        self.device_id = deviceID
        self.model_path = modelPath
        self.model_id = None
        self.context = None
        self.input_data = []
        self.output_data = []
        self.model_desc = None
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.result = []
        self.batch_size=batch_size
        try:
            self.init_resource()
        except:
            print("Initialize failed, something wrong with acl or your model")

    def init_resource(self):
        tic=time.time()
        ret = acl.init()#初始化
        check_ret("acl.init", ret)
        ret = acl.rt.set_device(self.device_id)#指定device
        check_ret("acl.rt.set_device", ret)
        self.context, ret = acl.rt.create_context(self.device_id)#显式创建context
        check_ret("acl.rt.create_context", ret)
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)#加载模型,返回模型的指针
        check_ret("acl.mdl.load_from_file", ret)
        self.model_desc = acl.mdl.create_desc()#创建desc数据的指针
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)#获取desc数据,存到model_desc所指的地址
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)#输入个数
        #print(f"input_size:{input_size}")
        output_size = acl.mdl.get_num_outputs(self.model_desc)#输出个数
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")
        toc=time.time()
        delta=toc-tic
        print(f"Initialization : {delta} s")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]#buffer_method在最前面定义
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)  
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size,ACL_MEM_MALLOC_HUGE_FIRST)
            # HUGE_FIRST不是数据大端存储的意思,是优先分配大页内存的意思
            check_ret("acl.rt.malloc", ret)
            if des == "in":
                self.input_data.append({"buffer": temp_buffer,"size": temp_buffer_size})
                print(f"Input (Byte):{temp_buffer_size}")
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer,"size": temp_buffer_size})
                print(f"Output (Byte):{temp_buffer_size}")

    def run(self, images):
        self._data_from_host_to_device(images)
        self.forward()
        self._destroy_databuffer()
        self._data_from_device_to_host()

    def _data_from_host_to_device(self, images):
        tic=time.time()
        # copy images to device
        self._data_interaction(images, ACL_MEMCPY_HOST_TO_DEVICE)
        #print("data_interaction h_to_d OK")
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        toc=time.time()
        delta=toc-tic
        print(f"\nhost to device: {delta} s")

    def _gen_dataset(self, type_str):
        dataset = acl.mdl.create_dataset()
        temp_dataset = None
        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        elif type_str == "out":
            self.load_output_dataset = dataset
            temp_dataset = self.output_data
        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def forward(self):
        tic=time.time()
        ret = acl.mdl.execute(self.model_id,
                              self.load_input_dataset,
                              self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        toc=time.time()
        delta=toc-tic
        print(f"forward: {delta} s")

    def _data_from_device_to_host(self):
        tic=time.time()
        res = []
        # copy device to host
        self._data_interaction(res, ACL_MEMCPY_DEVICE_TO_HOST)
        #print(f"res:{res}")
        dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, 0)
        #print(f"dims:{dims}")
        check_ret("acl.mdl.get_cur_output_dims", ret)
        out_dim = dims['dims']
        for temp in res:
            ptr = temp["buffer"]
            #float32类型的数据
            bytes_data = acl.util.ptr_to_bytes(ptr, temp["size"])
            data = np.frombuffer(bytes_data, dtype=np.float32).reshape(out_dim)
            self.result.append(data)
        #free host memory
        for item in res:
            ptr = item['buffer']
            ret = acl.rt.free_host(ptr)
            check_ret('acl.rt.free_host', ret)
        toc=time.time()
        delta=toc-tic
        print(f"device to host: {delta} s")

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def _data_interaction(self, dataset, policy):
        temp_data_buffer = self.input_data \
            if policy == ACL_MEMCPY_HOST_TO_DEVICE \
            else self.output_data
        if len(dataset) == 0 and policy == ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})
        #print(temp_data_buffer)
        #print(sys.getsizeof(dataset))
        for i, item in enumerate(temp_data_buffer):
            if policy == ACL_MEMCPY_HOST_TO_DEVICE:
                bytes_data = dataset[i].tobytes()
                #bytes_data = dataset[i].tobytes()
                ptr = acl.util.bytes_to_ptr(bytes_data)
                ret = acl.rt.memcpy(item["buffer"],
                                    item["size"],
                                    ptr,
                                    item["size"],
                                    policy)
                #item['buffer']是目标地址,ptr是源地址,host_to_device
                check_ret("acl.rt.memcpy", ret)
            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr,
                                    item["size"],
                                    item["buffer"],
                                    item["size"],
                                    policy)
                check_ret("acl.rt.memcpy", ret)
                #device_to_host

    def release_resource(self):
        tic=time.time()
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)
        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None
        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        ret = acl.finalize()
        check_ret("acl.finalize", ret)
        toc=time.time()
        delta=toc-tic
        print(f"release resources: {delta} s")      

    def print_result(self):
        print(self.result) 

    def get_result(self):
        return self.result


def callback_get_label(dataset, idx):
    return dataset[idx]["class"]


class EcgPipelineDataset1D(Dataset):
    def __init__(self, path, mode=128):
        super().__init__()
        record = wfdb.rdrecord(path)
        self.signal = None
        self.mode = mode
        for sig_name, signal in zip(record.sig_name, record.p_signal.T):
            if sig_name in ["MLII", "II"] and np.all(np.isfinite(signal)):
                self.signal = scale(signal).astype("float32")
        if self.signal is None:
            raise Exception("No MLII LEAD")

        self.peaks = find_peaks(self.signal, distance=180)[0]
        mask_left = (self.peaks - self.mode // 2) > 0
        mask_right = (self.peaks + self.mode // 2) < len(self.signal)
        mask = mask_left & mask_right
        self.peaks = self.peaks[mask]

    def __getitem__(self, index):
        peak = self.peaks[index]
        left, right = peak - self.mode // 2, peak + self.mode // 2

        img = self.signal[left:right]
        img = img.reshape(1, img.shape[0])

        return {"image": img, "peak": peak}

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        return data_loader

    def __len__(self):
        return len(self.peaks)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


class BasePipeline:
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get("exp_name", None)
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.res_dir = osp.join(self.config["exp_dir"], self.exp_name, "results")
        os.makedirs(self.res_dir, exist_ok=True)

        self.model = self._init_net()

        self.pipeline_loader = self._init_dataloader()

        self.mapper = json.load(open(config["mapping_json"]))
        self.mapper = {j: i for i, j in self.mapper.items()}

    def _init_net(self):
        raise NotImplemented

    def _init_dataloader(self):
        raise NotImplemented

    def run_pipeline(self):
        pd_class = np.empty(0)
        pd_peaks = np.empty(0)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.pipeline_loader)):
                # self.model=self._init_net()
                inputs = batch["image"] #返回batch,1,128的tensor
                inputs = inputs.numpy()
                batch_size = self.config["batch_size"]
                try:
                    inputs = inputs.astype(np.float32).reshape(batch_size,1,1,128)#1
                except:
                    current_shape = inputs.shape[0]
                    padding = np.zeros((batch_size,1,128), dtype=np.float32)
                    padding[:current_shape, :, :] = inputs
                    inputs = padding
                    inputs = inputs.astype(np.float32).reshape(batch_size,1,1,128)#1
                inputx = []
                inputx.append(inputs)
                #print("\n",inputx)
                self.model.run(inputx)
                predictions = self.model.get_result() #返回batch,8的tensor
                predictions=np.array(predictions).reshape(self.config["batch_size"],-1)
                #print(predictions)
                # predictions=torch.from_numpy(predictions)
                # classes = predictions.topk(k=1)[1].view(-1).numpy()
                classes = predictions.argmax(axis=1).reshape(-1)
                #print(classes)
                
                pd_class = np.concatenate((pd_class, classes))
                pd_peaks = np.concatenate((pd_peaks, batch["peak"]))
                # self.model.release_resource()
        
        tic=time.time()
        self.model.release_resource()
        pd_class = pd_class.astype(int)
        pd_peaks = pd_peaks.astype(int)

        annotations = []
        for label, peak in zip(pd_class, pd_peaks):
            #print(label,end=" ")
            label = self.mapper.get(label)
            #print(label,end=" ")
            if (
                label is not None
                and label != 'N'  #检查标签不是 'N'
                and peak < len(self.pipeline_loader.dataset.signal)
            ):
                annotations.append(
                    {
                        "x": peak,
                        "y": self.pipeline_loader.dataset.signal[peak],
                        "text": label,
                        "xref": "x",
                        "yref": "y",
                        "showarrow": True,
                        "arrowcolor": "black",
                        "arrowhead": 1,
                        "arrowsize": 2,
                    },
                )

        #print("\n",annotations)

        if osp.exists(self.config["ecg_data"] + ".atr"):
            ann = wfdb.rdann(self.config["ecg_data"], extension="atr")
            for label, peak in zip(ann.symbol, ann.sample):
                if peak < len(self.pipeline_loader.dataset.signal) and label != "N":
                    annotations.append(
                        {
                            "x": peak,
                            "y": self.pipeline_loader.dataset.signal[peak] - 0.1,
                            "text": label,
                            "xref": "x",
                            "yref": "y",
                            "showarrow": False,
                            "bordercolor": "#c7c7c7",
                            "borderwidth": 1,
                            "borderpad": 4,
                            "bgcolor": "#ffffff",
                            "opacity": 1,
                        },
                    )

        fig = go.Figure(
            data=go.Scatter(
                x=list(range(len(self.pipeline_loader.dataset.signal))),
                y=self.pipeline_loader.dataset.signal,
            ),
        )
        fig.update_layout(
            title="ECG",
            xaxis_title="Time",
            yaxis_title="ECG Output Value",
            title_x=0.5,
            annotations=annotations,
            autosize=True,
        )

        fig.write_html(
            osp.join(self.res_dir, osp.basename(self.config["ecg_data"] + ".html")),
        )
        toc=time.time()
        delta=toc-tic
        print(f"\nPost processing: {delta} s")


class Pipeline1D(BasePipeline):
    def __init__(self, config):
        super().__init__(config)

    def _init_net(self):
        model = Net(DEVICE_ID,self.config["om_model_path"],self.config["batch_size"])
        return model

    def _init_dataloader(self):
        inference_loader = EcgPipelineDataset1D(self.config["ecg_data"]).get_dataloader(
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )
        return inference_loader

if __name__ == "__main__":
    args = parse_args()
    config = json.loads(open(args.config).read())
    tic=time.time()
    pipeline = Pipeline1D(config)
    toc=time.time()
    delta=toc-tic
    print(f"\nPre-processing: {delta} s")
    pipeline.run_pipeline() 