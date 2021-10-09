import tensorflow as tf
import numpy as np
import json
import gc

import config
from Sregulizer import WStabilizer, BStabilizer
from Control_NetWork import Controller_Network
from custom_LSTM import LSTM_Network
from tqdm import tqdm


class LSTM_Task_Nerwork:
    def __init__(self, lstm_layers=[128], dense_layers=[64, 32, 2],
                 num_output_classes=2, nk=10, dynamic_coeff=0.001,
                 path_to_report='fully_trained.json', penalty=0.0001,
                 controller_epochs=20, num_controller_lstm=1):
        self.weights = {}  # we will preserve our parameters in this dictionary of ndarrays through time
        self.T = 1
        self.lstm_layers = lstm_layers
        self.number_of_lstms = len(lstm_layers)
        self.number_of_dense_layers = len(dense_layers)
        self.dense_layers = dense_layers
        self.num_output_classes = num_output_classes
        self.lstm_network = None  # we will use a fix lstm network and alter the dense network with incoming data
        if num_output_classes == 2:
            self.mode = 'binary'
        else:
            self.mode = 'multi-class'
        # weight below this value will be considered as zero
        self.nk = nk
        self.dynamic_coeff = dynamic_coeff
        self.reports = []  # array of jsons
        self.path_to_report = path_to_report
        self.penalty = penalty
        self.controller_epochs = controller_epochs
        self.num_controller_lstm = num_controller_lstm

    def initialize_weights(self, path="base-lstm",
                           input_model=None):  # loads the weights from our base model to the weights
        if input_model is None:
            reconstructed_model = LSTM_Network(input_size=config.pkt_size, LSTM_units=self.lstm_layers,
                                               dense_units=self.dense_layers)
            reconstructed_model.load_weights(path)
        else:
            reconstructed_model = input_model
        # getting the weights for our cov2d layers
        # for i in range(self.number_of_conv2ds):
        #     number_of_filters = reconstructed_model.layers[i].trainable_variables[0].shape[-1]
        #     number_of_channels = reconstructed_model.layers[i].trainable_variables[0].shape[-2]
        #     self.weights['conv' + str(i)] = reconstructed_model.layers[i].trainable_variables[0].numpy().\
        #         reshape(number_of_filters,number_of_channels,self.kernel_size,self.kernel_size)
        #     self.weights['bconv' + str(i)] = reconstructed_model.layers[i].trainable_variables[1].numpy() # biases
        #
        # self.cnn_network = self.build_cnn_model()
        self.lstm_network = reconstructed_model.lstm

        # getting the weights for dense layers
        for i in range(self.number_of_dense_layers):  # the + 1 is for flatten layer
            self.weights['dense' + str(i)] = reconstructed_model.dense_model.layers[i].trainable_variables[0].numpy()
            self.weights['bdense' + str(i)] = reconstructed_model.dense_model.layers[i].trainable_variables[
                1].numpy()  # biases

    def save_dense_weights(self, model):
        weight_layers_in_model = [3 * (i + 1) for i in range(self.number_of_dense_layers - 1)]
        weight_layers_in_model.append(len(model.layers) - 1)
        for i in range(self.number_of_dense_layers):  # the + 1 is for flatten layer
            # self.number_of_conv2ds + 1, self.number_of_conv2ds + 1 + self.number_of_dense_layers + 1
            index_of_weight = weight_layers_in_model[i]
            if i == self.number_of_dense_layers - 1:

                self.weights['dense' + str(i)] = model.layers[index_of_weight].weights[0].numpy()
                self.weights['bdense' + str(i)] = model.layers[index_of_weight].weights[1].numpy()  # biases
            else:
                self.weights['dense' + str(i)] = np.concatenate((model.layers[index_of_weight - 2].weights[0].numpy(),
                                                                 model.layers[index_of_weight - 1].weights[0].numpy()),
                                                                axis=1)
                self.weights['bdense' + str(i)] = np.concatenate((model.layers[index_of_weight - 2].weights[1].numpy(),
                                                                  model.layers[index_of_weight - 1].weights[1].numpy()),
                                                                 axis=0)  # biases

    def build_complete_model(self):
        lstm_network = LSTM_Network(input_size=config.pkt_size, LSTM_units=self.lstm_layers,
                                    dense_units=self.dense_layers)
        lstm_network.lstm = self.lstm_network
        dense_layers = [tf.keras.Input(shape=(self.lstm_layers[-1],))]
        for i in range(self.number_of_dense_layers - 1):
            kernels = tf.keras.backend.variable(self.weights['dense' + str(i)])
            biases = tf.keras.backend.variable(self.weights['bdense' + str(i)])
            dense_prev = tf.keras.layers.Dense(self.weights['dense' + str(i)].shape[1],
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            dense_prev.trainable = False
            dense_layers.append(dense_prev)
        index_of_last = self.number_of_dense_layers - 1
        kernel = tf.keras.backend.variable(
            self.weights['dense' + str(index_of_last)])
        bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
        output_layer = tf.keras.layers.Dense(self.num_output_classes, activation=tf.keras.activations.sigmoid,
                                             kernel_initializer=tf.keras.initializers.Constant(kernel),
                                             bias_initializer=tf.keras.initializers.Constant(bias))
        dense_layers.append(output_layer)
        lstm_network.dense_model = tf.keras.Sequential(dense_layers)
        return lstm_network

    def build_expanded_dense_netowrk(self, actions=[]):
        inputs = tf.keras.Input(shape=(self.lstm_layers[-1],))
        input_of_dense = inputs
        for i in range(self.number_of_dense_layers - 1):
            if i == 0:
                kernels = tf.keras.backend.variable(self.weights['dense' + str(i)])
            else:
                kernels = tf.keras.backend.variable(
                    self.add_dense_weight(self.weights['dense' + str(i)], actions[i - 1]))
            biases = tf.keras.backend.variable(self.weights['bdense' + str(i)])
            dense_prev = tf.keras.layers.Dense(self.weights['dense' + str(i)].shape[1],
                                               activation=tf.keras.activations.relu,
                                               kernel_initializer=tf.keras.initializers.Constant(kernels),
                                               bias_initializer=tf.keras.initializers.Constant(biases)
                                               )
            dense_prev.trainable = False
            dense_new = tf.keras.layers.Dense(actions[i], activation=tf.keras.activations.relu,
                                              )
            input_of_dense = tf.keras.layers.Concatenate(axis=-1)(
                [dense_prev(input_of_dense), dense_new(input_of_dense)])
            # new
            # input_of_dense = tf.keras.layers.BatchNormalization()(input_of_dense)
            # input_of_dense = tf.keras.layers.Dropout(0.3)(input_of_dense)

        index_of_last = self.number_of_dense_layers - 1
        # if verbose:
        #     print("build " + str(index_of_last))
        number_of_nodes = self.weights['dense' + str(index_of_last)].shape[1]
        number_of_lower_nodes = self.weights['dense' + str(index_of_last)].shape[0]
        print(actions)
        if actions[-1] != 0:
            kernel = tf.keras.backend.variable(
                self.add_dense_weight(self.weights['dense' + str(index_of_last)], actions[-1]))
            bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
            wstabilzer = WStabilizer(
                selected_lower_nodes=[i for i in range(number_of_lower_nodes, number_of_lower_nodes + actions[-1])],
                selected_upper_nodes=[0, 1],
                num_of_all_lower_nodes=number_of_lower_nodes + actions[-1],
                num_of_all_upper_nodes=self.weights['dense' + str(index_of_last)].shape[1]
            )
            wstabilzer.configure_weight_matrix(
                self.add_dense_weight(self.weights['dense' + str(index_of_last)], actions[-1]))

            output_layer_prev = tf.keras.layers.Dense(self.num_output_classes, activation=tf.keras.activations.sigmoid,
                                                      kernel_regularizer=wstabilzer, kernel_initializer=
                                                      tf.keras.initializers.Constant(kernel),
                                                      bias_initializer=tf.keras.initializers.Constant(bias))(
                input_of_dense)
        else:
            kernel = tf.keras.backend.variable(self.weights['dense' + str(index_of_last)])
            bias = tf.keras.backend.variable(self.weights['bdense' + str(index_of_last)])
            output_layer_prev = tf.keras.layers.Dense(self.num_output_classes, activation=tf.keras.activations.sigmoid,
                                                      kernel_initializer=
                                                      tf.keras.initializers.Constant(kernel),
                                                      bias_initializer=tf.keras.initializers.Constant(bias))
            output_layer_prev.trainable = False
            output_layer_prev = output_layer_prev(input_of_dense)

        return tf.keras.Model(inputs=inputs, outputs=output_layer_prev, name='expanded_model' + str(self.T))

    def add_dense_weight(self, layer_weights, number_of_new_nodes):
        if number_of_new_nodes == 0:
            return layer_weights
        new_weights = np.array([[0 for i in range(layer_weights.shape[1])] for j in range(number_of_new_nodes)])
        return np.concatenate((layer_weights, new_weights))

    def evaluate(self, test_flows, test_labels, model=None, with_lstm=False):
        if with_lstm:
            if model is None:
                model = self.build_complete_model()
            test_flows = tf.convert_to_tensor(test_flows, dtype=tf.float32)
        else:
            # if model is None:
            #     model = self.create_dense_network()
            lstm_model = self.lstm_network
            test_flows = tf.convert_to_tensor(test_flows, dtype=tf.float32)
            states = lstm_model.get_initial_state(test_flows)

            outputs = None
            for i in range(test_flows.shape[1]):
                outputs, states = lstm_model(test_flows[:, i], states)

            test_flows = tf.convert_to_tensor(outputs, dtype=tf.float32)

        labels = test_labels
        output = model(test_flows)
        if with_lstm:
            model.reset_state()
        predicted_outputs = tf.argmax(output, axis=1)
        number_of_corrects = 0
        number_of_samples = len(labels)
        for i in range(len(predicted_outputs)):
            if predicted_outputs[i] == labels[i]:
                number_of_corrects += 1
        del test_flows
        gc.collect()
        return number_of_corrects / number_of_samples

    def reward(self, accuracy, actions):
        return accuracy - self.penalty * sum(actions)

    def add_flows(self, train_flows, train_labels, validation_flows, validation_labels, test_flows, test_labels,
                  train_attacks, target_attacks, previous_flows, previous_labels, path, report=True):

        train_flows = tf.convert_to_tensor(train_flows, dtype=tf.float32)
        finals = []
        states = self.lstm_network.get_initial_state(train_flows)
        outputs = None
        for i in range(train_flows.shape[1]):
            outputs, states = self.lstm_network(train_flows[:, i], states)

        train_flows = tf.convert_to_tensor(outputs, dtype=tf.float32)

        best_actions = ''
        best_model = None
        best_reward = 0
        best_validation_acc = 0
        rewards = []
        all_actions = []
        # we want to check if we undergo catastrophic forgetting
        initial_accuracy_on_original_data = self.evaluate(previous_flows, previous_labels, model=None, with_lstm=True)
        initial_accuracy = self.evaluate(test_flows, test_labels, model=None, with_lstm=True)
        for iter in range(3):
            if best_validation_acc > 0.9:
                break
            for batch_size in config.task_batch_sizes:
                for learning_rate in [1e-6, 1e-3, 1]:
                    controller_network = Controller_Network(state_space=self.nk,
                                                            number_of_actions=self.number_of_dense_layers - 1,
                                                            lstm_num_layers=self.num_controller_lstm)
                    for reinforcement_epoch in range(self.controller_epochs):
                        print('reinforcement epoch : ', reinforcement_epoch)

                        actions = controller_network.get_actions()
                        expanded_dense_network = self.build_expanded_dense_netowrk(actions)
                        # expanded_dense_network.summary()
                        # expanded_dense_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                        # loss='sparse_categorical_crossentropy',
                        # metrics=['accuracy'])
                        # expanded_dense_network.fit(train_flows,train_labels,batch_size = config.new_task_batch_size,epochs = config.task_epochs)

                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # define our optimizer
                        for epoch in range(config.task_epochs):

                            print("epoch : ", epoch + 1)
                            for idx in tqdm(range(0, 256 // batch_size, batch_size)):
                                flows = train_flows[idx: idx + batch_size]
                                labels = train_labels[idx: idx + batch_size]

                                with tf.GradientTape() as tape:
                                    predicted_outputs = expanded_dense_network(flows)
                                    loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels,
                                                                                                  predicted_outputs)
                                grads = tape.gradient(loss_value, expanded_dense_network.trainable_variables)
                                # print(grads[-2])
                                optimizer.apply_gradients(zip(grads, expanded_dense_network.trainable_variables))
                        accuracy_val = self.evaluate(validation_flows, validation_labels, expanded_dense_network)
                        print(accuracy_val)
                        print("****")
                        reward = self.reward(accuracy_val, actions)
                        rewards.append(reward)
                        all_actions.append(actions)
                        if best_reward <= reward:
                            best_reward = reward
                            best_model = expanded_dense_network
                            best_validation_acc = accuracy_val
                            best_actions = actions
                        else:
                            del expanded_dense_network
                            gc.collect()

                        controller_network.train(reward)
        # print(self.weights['dense2'].shape)
        # print(self.weights['dense2'])
        # print("****************")
        # print(best_model.layers[-1].weights[0].shape)
        # print(best_model.layers[-1].weights[0])
        test_acc = self.evaluate(test_flows, test_labels, model=best_model)
        self.save_dense_weights(best_model)
        test_acc = self.evaluate(test_flows, test_labels, with_lstm=True)
        print("validation accuracy of best model: ", best_validation_acc)
        print("test accuracy of best model: ", test_acc)
        # print(len(previous_flows))
        later_accuracy_on_original = self.evaluate(previous_flows, previous_labels, with_lstm=True)
        print(actions)
        print(rewards)
        if report:
            self.report_model(test_flows, test_labels, best_actions, train_attacks, target_attacks
                              , path=path, initial_accuracy=initial_accuracy,
                              initial_accuracy_on_previous=initial_accuracy_on_original_data,
                              later_accuracy_on_previous=later_accuracy_on_original)

        return test_acc

    def report_model(self, test_flows, test_labels, actions, train_attacks, target_attacks, path, initial_accuracy,
                     initial_accuracy_on_previous,
                     later_accuracy_on_previous):
        model = self.build_complete_model()
        test_flows = tf.convert_to_tensor(test_flows, dtype=tf.float32)
        labels = test_labels
        output = model.predict(test_flows)
        model.reset_state()
        predicted_outputs = tf.argmax(output, axis=1)
        number_of_samples = len(labels)
        output = ''
        if self.mode == 'binary':
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(len(predicted_outputs)):
                if predicted_outputs[i] == labels[i]:
                    if labels[i] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if labels[i] == 1:
                        fn += 1
                    else:
                        fp += 1
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            try:
                precision = tp / (tp + fp)
            except:
                precision = 0
            try:
                recall = tp / (tp + fn)
            except:
                recall = 0
            try:
                f1_score = (2 * recall * precision) / (recall + precision)
            except:
                f1_score = 0
            accuracy = round(accuracy, 2)
            precision = round(precision, 2)
            recall = round(recall, 2)
            f1_score = round(f1_score, 2)
            output += 'Type : ' + self.mode + '\n'
            classes_trained = ''
            for classes in train_attacks:
                classes_trained += classes + ' '
            output += 'Flows trained on : ' + classes_trained + '\n'

            classes_tested = ''
            for classes in target_attacks:
                classes_tested += classes + ' '

            output += 'Flows targeted : ' + classes_tested + '\n'
            output += 'Number of Flows tested on : ' + str(number_of_samples) + '\n'
            output += "accuracy on original data before training : " + str(initial_accuracy_on_previous) + '\n'
            output += "accuracy on original data after training : " + str(later_accuracy_on_previous) + '\n'
            output += "accuracy before training : " + str(initial_accuracy) + '\n'
            nodes_added = ''
            for action in actions:
                nodes_added += ' ' + str(action)
            output += 'Number of nodes added to each layer : ' + nodes_added + '\n'
            output += '  Precision     Recall    F1-score\n'
            output += '#####################################\n'
            output += '  ' + str(precision) + '     ' + str(recall) + '     ' + str(f1_score)
        with open(path, 'w') as results:
            results.write(output)









