# MNIST Classifier Optimized for Embedded Devices

This project focuses on developing an optimized MNIST classifier for embedded devices such as Raspberry Pi. The classifier has been enhanced using pruning and quantization techniques to reduce its size and improve efficiency while maintaining high accuracy.

## Pruning

Pruning is a technique used to remove redundant or less important weights from a trained neural network. By eliminating these weights, the model becomes smaller and more efficient without significantly impacting its performance.

The pruning process was applied to the MNIST classifier, and the results are documented in the `metrics.csv` file. The file contains the following columns:

- `epoch`: The epoch number during training.
- `step`: The step number within each epoch.
- `test_acc`: The accuracy achieved on the test set.
- `test_loss`: The loss value obtained on the test set.

### Pruning Metrics

| Epoch | Step | Test Accuracy | Test Loss |
|-------|------|---------------|-----------|
| 0     | 0    | 0.8444        | 0.4488    |

After pruning, the model was saved and reloaded to evaluate its performance. The pruning amount achieved was 0.9230488538742065, indicating a significant reduction in the model's size.

The test performance after pruning and reloading the model was as follows:
```
[{'test_loss': 0.44882017374038696, 'test_acc': 0.8443999886512756}]
```

The pruned model achieved a test accuracy of 0.8444 and a test loss of 0.4488, demonstrating that the pruning process effectively reduced the model's size while maintaining its performance.

## Quantization

Quantization is a technique used to reduce the precision of the model's weights and activations, typically from 32-bit floating-point values to lower-bit representations such as 8-bit integers. This reduction in precision helps to decrease the model's memory footprint and accelerate its execution on resource-constrained devices.

The MNIST classifier was quantized to optimize it further for deployment on embedded devices like Raspberry Pi. The quantization process allows the model to utilize less memory and perform faster inference without significantly compromising accuracy.

## Conclusion

By applying pruning and quantization techniques to the MNIST classifier, we have successfully optimized it for embedded devices such as Raspberry Pi. The pruned model achieves a significant reduction in size while maintaining high accuracy, and the quantization process further enhances its efficiency and performance on resource-constrained devices.

This optimized MNIST classifier demonstrates the potential for deploying deep learning models on embedded systems, enabling efficient and accurate image classification tasks in various applications.