import numpy as np
from bokeh.plotting import figure, output_file, save  # , show

from tensorboard.backend.event_processing import event_accumulator

file_path = "/tigress/alexeys/worked_Graphs/Graph16_momSGD_new/"
ea1 = event_accumulator.EventAccumulator(
    file_path + "events.out.tfevents.1502649990.tiger-i19g10")
ea1.Reload()

ea2 = event_accumulator.EventAccumulator(
    file_path + "events.out.tfevents.1502652797.tiger-i19g10")
ea2.Reload()

histograms = ea1.Tags()['histograms']
# ages': [], 'audio': [], 'histograms': ['input_2_out',
# 'time_distributed_1_out', 'lstm_1/kernel_0', 'lstm_1/kernel_0_grad',
# 'lstm_1/recurrent_kernel_0', 'lstm_1/recurrent_kernel_0_grad',
# 'lstm_1/bias_0', 'lstm_1/bias_0_grad', 'lstm_1_out', 'dropout_1_out',
# 'lstm_2/kernel_0', 'lstm_2/kernel_0_grad', 'lstm_2/recurrent_kernel_0',
# 'lstm_2/recurrent_kernel_0_grad', 'lstm_2/bias_0', 'lstm_2/bias_0_grad',
# 'lstm_2_out', 'dropout_2_out', 'time_distributed_2/kernel_0',
# 'time_distributed_2/kernel_0_grad', 'time_distributed_2/bias_0',
# 'time_distributed_2/bias_0_grad', 'time_distributed_2_out'], 'scalars':
# ['val_roc', 'val_loss', 'train_loss'], 'distributions': ['input_2_out',
# 'time_distributed_1_out', 'lstm_1/kernel_0', 'lstm_1/kernel_0_grad',
# 'lstm_1/recurrent_kernel_0', 'lstm_1/recurrent_kernel_0_grad',
# 'lstm_1/bias_0', 'lstm_1/bias_0_grad', 'lstm_1_out', 'dropout_1_out',
# 'lstm_2/kernel_0', 'lstm_2/kernel_0_grad', 'lstm_2/recurrent_kernel_0',
# 'lstm_2/recurrent_kernel_0_grad', 'lstm_2/bias_0', 'lstm_2/bias_0_grad',
# 'lstm_2_out', 'dropout_2_out', 'time_distributed_2/kernel_0',
# 'time_distributed_2/kernel_0_grad', 'time_distributed_2/bias_0',
# 'time_distributed_2/bias_0_grad', 'time_distributed_2_out'], 'tensors':
# [], 'graph': True, 'meta_graph': True, 'run_metadata': []}

for h in histograms:
    x1 = np.array(ea1.Histograms(h)[0].histogram_value.bucket_limit[:-1])
    y1 = ea1.Histograms(h)[0].histogram_value.bucket[:-1]
    x2 = np.array(ea2.Histograms(h)[0].histogram_value.bucket_limit[:-1])
    y2 = ea2.Histograms(h)[0].histogram_value.bucket[:-1]

    h = h.replace("/", "_")

    p = figure(title=h, y_axis_label="Arbitrary units",
               x_axis_label="Arbitrary units")
    #           , y_axis_type="log")

    p.line(x1, y1, legend="float16, SGD with momentum",
           line_color="green", line_width=2)

    p.line(x2, y2, legend="float32, SGD with momentum",
           line_color="indigo", line_width=2)

    p.legend.location = "top_right"

    output_file("plot" + h + ".html", title=h)
    save(p)  # open a browser
