To configure the project you can run the `configure.sh` file, 
which will create build directories `debug` and `release`.

To build, simply run

```bash
cmake --build [debug|release]
```

From the root project directory

# Usage:

Each neural network is held in a directory, containing 4 principle files.

```
[network]/
          .
          ..
          config.txt
          topology.txt
          training_examples.txt
          weights.bin
```

`config.txt`: Text file containing hyperparameters for the network, in this order:

`top_learning_rate bot_learning_rate decay_rate learning_rate_cycle_length hidden_unit_activation output_unit_activation`

Supported activation functions are `sigmoid`, `tanh`, `binary` and `none`

`topology.txt`: Text file containing the size of each layer in order, starting with the input layer, and finishing with the output layer.


`training_examples.txt`: Text file containing training examples to be used, a delineated file of floating point numbers, with the input vector followed by the expected output vector for each example.
> ie for the xor dataset the file could look like:
> ```
> 0 0 0
> 0 1 1
> 1 0 1
> 1 1 0
> ```

Finally, `weights.bin` olds any saved weights for this particular networks.

## Commands:

`exit`: exit the program.

`train <epoch> <output csv>`:  train the neural network on the dataset for `<epoch>` epochs, and write statistics to a csv file.

`save`: save weights from ram to disk. (If you don't want to overwrite, you have to rename the old weights file as a backup)

`test <test file> <output csv>` Test, followed by the input vector to manually test the program, writes the output to stdout.


## To run the given networks:

To run a network yourself, once the binary is built, just execute and pass the root directory of a network.

For example, you can first run the following to build the release binary;

`$ cmake --build release`

Then run the arithmetic network for testing:

`$ release/main networks/arithmetic`

