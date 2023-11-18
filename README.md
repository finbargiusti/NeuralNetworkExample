# COMP30230 Finbar Giusti (21372821) Neural Network assignment 

To build the Makefiles:

```bash
cmake src/;
```

To build the `main` executable:

```bash
make
```

# Usage:

Each neural network is held in a directory, containing 3 principle files.

```
[network]/
          .
          ..
          topology.txt
          training_examples.txt
          weights.bin
```

`toplogy.txt`: Text file containing the size of each layer in order, starting with the input layer, and finishing with the output layer.


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

`train <epoch>`:  train the neural network on the dataset for `<epoch>` epochs.
> Coming soon: automatic training data collection

`save`: save weights from ram to disk. (If you don't want to overwrite, you have to rename the old weights file as a backup)

`test [inputs]` Test, followed by the input vector to manually test the program, writes the output to stdout.
> Coming soon: automatic testing

`rate <rate>` Change the learning rate to `<rate>`


``
