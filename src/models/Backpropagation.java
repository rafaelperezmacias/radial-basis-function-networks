package models;

import utils.*;

import java.util.*;

public class Backpropagation extends Regression {

    public static final int SIGMOID_FUNCTION = 1;
    public static final int TANH_FUNCTION = 2;
    public static final int RELU_FUNCTION = 3;

    private static final int[] FUNCTIONS = {
            SIGMOID_FUNCTION,
            TANH_FUNCTION,
            RELU_FUNCTION
    };

    public static final int BATCH_GRADIENT = 1;
    public static final int MINI_BATCH_GRADIENT = 2;
    public static final int STOCHASTIC_GRADIENT = 3;

    private static final int[] GRADIENTS = {
            BATCH_GRADIENT,
            MINI_BATCH_GRADIENT,
            STOCHASTIC_GRADIENT
    };

    private Backpropagation()
    {

    }

    public static Model generateModel(DataSet dataSet, Params params) throws Exception {
        Backpropagation backpropagation = new Backpropagation();
        return backpropagation.makeModel(dataSet, params);
    }

    @Override
    protected Model makeModel(DataSet dataSet) throws Exception {
        throw new Exception("Not supported");
    }

    @Override
    protected Model makeModel(DataSet dataSet, AbstractParams params) throws Exception {
        validateDataSetAndParams(dataSet, params);
        int idxTarget = DataSetFunctions.getIndexAtAtributeFromDataSet(dataSet.getHeaders(), dataSet.getTarget());
        // Conjunto de datos a trabajar
        HashMap<String, double[]> targets = new HashMap<>();
        double[][] X = DataSetFunctions.generateNumericMatrix(dataSet, getColumns(dataSet, idxTarget));
        double[][] Y = getTargetsFromTheDataSet(dataSet, idxTarget, targets);
        // Calculate step
        int step = getStepFromBatch((Params) params, X);
        // Setup of layers
        ArrayList<Layer> layers = new ArrayList<>();
        setupLayers(layers, dataSet, (Params) params, targets);
        // Desempaquetado de parametros y variables auxiliares
        double learningRate = ((Params) params).learningRate;
        int epochs = ((Params) params).epochs;
        int epoch = 0;
        double minError = ((Params) params).minError;
        double error = Double.MAX_VALUE;
        // Empezamos el entrenamiento
        while ( error > minError && epoch < epochs ) {
            // Iteraciones por epoca
            error = 0;
            for ( int i = 0; i < X.length; ) {
                // Batch de trabajo
                int j = 0;
                for ( ; j < step && i < X.length; i++, j++ ) {
                    // Forward
                    initFirstLayer(layers.get(0), X[i]);
                    for ( int k = 1; k < layers.size(); k++ ) {
                        layers.get(k).netActivations = AlgebraicHelpers.matrixMultiplicationByVector(layers.get(k).weights, layers.get(k - 1).outputs);
                        updateOutputsForForward(layers.get(k));
                    }
                    // Error por instancia
                    double result = 0;
                    for ( int l = 0; l < Y[i].length; l++ ) {
                        result += Math.pow(Y[i][l] - layers.get(layers.size() - 1).outputs[l], 2);
                    }
                    error += Math.sqrt(result);
                    // Backpropagation
                    for ( int k = layers.size() - 1; k >= 1; k-- ) {
                        if ( layers.get(k).last ) {
                            layers.get(k).sensivilities = AlgebraicHelpers.vectorMultiplicationByNumber(
                                    AlgebraicHelpers.getHadamardProduct(
                                            getVectorOfDerivativeNetsForMLayer(layers.get(k)),
                                            AlgebraicHelpers.getVectorSubtraction(Y[i], layers.get(k).outputs)
                                    ), -2
                            );
                        } else {
                            layers.get(k).sensivilities = AlgebraicHelpers.matrixMultiplicationByVector(
                                    AlgebraicHelpers.matrixMultiplication(
                                            getDerivativeDiagonalMatrix(layers.get(k)),
                                            AlgebraicHelpers.transpose(getWeightsForBackpropagation(layers.get(k + 1)))
                                    ),
                                    layers.get(k + 1).sensivilities
                            );
                        }
                    }
                    // Gradient
                    for ( int k = 1; k < layers.size(); k++ ) {
                        layers.get(k).gradient  = AlgebraicHelpers.matrixSummation(
                                layers.get(k).gradient,
                                AlgebraicHelpers.matrixMultiplicationByNumber(
                                        AlgebraicHelpers.matrixMultiplication(
                                                vectorToMatrix(layers.get(k).sensivilities),
                                                AlgebraicHelpers.transpose(vectorToMatrix(layers.get(k - 1).outputs))
                                        ), -learningRate
                                )
                        );
                    }
                }
                // Update weigths
                for ( int k = 1; k < layers.size(); k++ ) {
                    layers.get(k).weights = AlgebraicHelpers.matrixSummation(
                            layers.get(k).weights,
                            AlgebraicHelpers.matrixDivisionByNumber(layers.get(k).gradient, j)
                    );
                    resetMatrixGradient(layers.get(k).gradient);
                }

            }
            epoch++;
        }
        // Modelo obtenido
        return new Model(dataSet, layers, targets, minError, error, epochs, epoch);
    }

    private static void initFirstLayer(Layer layer, double[] instance) {
        System.arraycopy(instance, 0, layer.outputs, 1, instance.length);
    }

    private void resetMatrixGradient(double[][] gradient) {
        for ( int i = 0; i < gradient.length; i++ ) {
            for ( int j = 0; j < gradient[0].length; j++ ) {
                gradient[i][j] = 0;
            }
        }
    }

    private double[][] vectorToMatrix(double[] vector) {
        double[][] result = new double[vector.length][1];
        for ( int i = 0; i < vector.length; i++ ) {
            result[i][0] = vector[i];
        }
        return result;
    }

    private double[][] getWeightsForBackpropagation(Layer layer) {
        double[][] result = new double[layer.weights.length][layer.weights[0].length - 1];
        for ( int i = 0; i < result.length; i++ ) {
            System.arraycopy(layer.weights[i], 1, result[i], 0, result[i].length);
        }
        return result;
    }

    private double[][] getDerivativeDiagonalMatrix(Layer layer) {
        double[][] result = new double[layer.neurons][layer.neurons];
        for ( int i = 0; i < result.length; i++ ) {
            for ( int j = 0; j < result[i].length; j++ ) {
                if ( i == j ) {
                    if ( layer.function == SIGMOID_FUNCTION ) {
                        result[i][j] = Functions.derivativeSigmoid(layer.netActivations[i]);
                    } else if ( layer.function == RELU_FUNCTION ) {
                        result[i][j] = Functions.derivativeRelu(layer.netActivations[i]);
                    } else if ( layer.function == TANH_FUNCTION ) {
                        result[i][j] = Functions.derivativeTanh(layer.netActivations[i]);
                    }
                } else {
                    result[i][j] = 0;
                }
            }
        }
        return result;
    }

    private double[] getVectorOfDerivativeNetsForMLayer(Layer layer) {
        double[] result = new double[layer.netActivations.length];
        for ( int i = 0; i < layer.netActivations.length; i++ ) {
            if ( layer.function == SIGMOID_FUNCTION ) {
                result[i] = Functions.derivativeSigmoid(layer.netActivations[i]);
            } else if ( layer.function == RELU_FUNCTION ) {
                result[i] = Functions.derivativeRelu(layer.netActivations[i]);
            } else if ( layer.function == TANH_FUNCTION ) {
                result[i] = Functions.derivativeTanh(layer.netActivations[i]);
            }
        }
        return result;
    }

    private static void updateOutputsForForward(Layer layer) {
        for ( int i = 0; i < layer.netActivations.length; i++ ) {
            if ( layer.last ) {
                if ( layer.function == SIGMOID_FUNCTION ) {
                    layer.outputs[i] = Functions.sigmoid(layer.netActivations[i]);
                } else if ( layer.function == RELU_FUNCTION ) {
                    layer.outputs[i] = Functions.relu(layer.netActivations[i]);
                } else if ( layer.function == TANH_FUNCTION ) {
                    layer.outputs[i] = Functions.tanh(layer.netActivations[i]);
                }
            } else {
                if ( layer.function == SIGMOID_FUNCTION ) {
                    layer.outputs[i + 1] = Functions.sigmoid(layer.netActivations[i]);
                } else if ( layer.function == RELU_FUNCTION ) {
                    layer.outputs[i + 1] = Functions.relu(layer.netActivations[i]);
                } else if ( layer.function == TANH_FUNCTION ) {
                    layer.outputs[i + 1] = Functions.tanh(layer.netActivations[i]);
                }
            }
        }
    }

    private void setupLayers(ArrayList<Layer> layers, DataSet dataSet, Params params, HashMap<String, double[]> targets) {
        // First layer
        Layer firstLayer = new Layer(0, 0);
        setupFirstLayer(firstLayer, dataSet);
        layers.add(firstLayer);
        // Hidden layers
        if ( params.hiddenLayers != null ) {
            layers.addAll(params.hiddenLayers);
            Layer lastLayer = firstLayer;
            for ( int i = 1; i < layers.size(); i++ ) {
                setupHiddenLayer(layers.get(i), lastLayer);
                lastLayer = layers.get(i);
            }
        }
        // Last layer
        Layer lastLayer = new Layer(targets.size(), params.lastNeuronFunctionActivation, true);
        setupLastLayer(lastLayer, layers.get(layers.size() - 1));
        layers.add(lastLayer);
    }

    private void setupLastLayer(Layer lastLayer, Layer prevLayer) {
        lastLayer.weights = new double[lastLayer.neurons][prevLayer.outputs.length];
        fillWeights(lastLayer.weights);
        lastLayer.gradient = new double[lastLayer.neurons][prevLayer.outputs.length];
        lastLayer.outputs = new double[lastLayer.neurons];
        lastLayer.sensivilities = new double[lastLayer.neurons];
        lastLayer.netActivations = new double[lastLayer.neurons];
    }

    private void setupHiddenLayer(Layer hiddenLayer, Layer lastLayer) {
        hiddenLayer.weights = new double[hiddenLayer.neurons][lastLayer.outputs.length];
        fillWeights(hiddenLayer.weights);
        hiddenLayer.gradient = new double[hiddenLayer.neurons][lastLayer.outputs.length];
        hiddenLayer.outputs = new double[hiddenLayer.neurons + 1];
        hiddenLayer.sensivilities = new double[hiddenLayer.neurons];
        hiddenLayer.netActivations = new double[hiddenLayer.neurons];
        hiddenLayer.outputs[0] = -1;
    }

    private void setupFirstLayer(Layer layer, DataSet dataSet) {
        layer.outputs = new double[dataSet.getHeaders().size()];
        layer.outputs[0] = -1;
    }

    private void fillWeights(double[][] weights) {
        for ( int i = 0; i < weights.length; i++ ) {
            for ( int j = 0; j < weights[i].length; j++ ) {
                weights[i][j] = getRandom();
            }
        }
    }

    private double getRandom() {
        int random = (int) (Math.random() * (100));
        int sign = (int) (Math.random() * 10);
        return (double) ((sign % 2 == 0) ? -random : random) / 100;
    }

    private int getStepFromBatch(Params params, double[][] X) {
        if ( params.gradient == STOCHASTIC_GRADIENT ) {
           return 1;
        }
        if ( params.gradient == MINI_BATCH_GRADIENT ) {
            return params.batchSize;
        }
        return X.length;
    }

    private static int[] getColumns(DataSet dataSet, int idxExcept) {
        int[] columns = new int[dataSet.getHeaders().size() - 1];
        for ( int i = 0, j = 0; i < dataSet.getHeaders().size(); i++ ) {
            if ( i == idxExcept ) {
                continue;
            }
            columns[j++] = i;
        }
        return columns;
    }

    private static double[][] getTargetsFromTheDataSet(DataSet dataSet, int idxTarget, HashMap<String, double[]> targets) {
        // Targets
        Set<String> targetsFromDataSet = new HashSet<>();
        for ( int i = 0; i < dataSet.getInstances().size(); i++ ) {
            targetsFromDataSet.add( dataSet.getInstances().get(i).get(idxTarget) );
        }
        List<String> targetsList = new ArrayList<>(targetsFromDataSet);
        // Transforma el target de 0,1,2,..., etc, a [1, 0, 0, 0], [0, 1, 0, 0]
        double[][] matrixTargets = new double[targetsList.size()][targetsList.size()];
        for ( int i = 0; i < targetsList.size(); i++ ) {
            double[] newTarget = new double[matrixTargets.length];
            Arrays.fill(newTarget, 0);
            newTarget[i] = 1;
            matrixTargets[i] = newTarget;
            targets.put(targetsList.get(i), newTarget);
        }
        // Actualiza los targets originales con los nuevos anteriormente generados
        double[][] targetsOut = new double[dataSet.getInstances().size()][targetsList.size()];
        for ( int i = 0; i < dataSet.getInstances().size(); i++ ) {
            int idx = 0;
            for ( int j = 0; j < targetsList.size(); j++ ) {
                if ( targetsList.get(j).equals(dataSet.getInstances().get(i).get(idxTarget)) ) {
                    idx = j;
                    break;
                }
            }
            System.arraycopy(matrixTargets[idx], 0, targetsOut[i],0, targetsOut[0].length);
        }
        return targetsOut;
    }

    private static void validateDataSetAndParams(DataSet dataSet, AbstractParams params) throws Exception {
        // DataSet
        DataSetFunctions.validateDataSet(dataSet);
        int idxTarget = DataSetFunctions.getIndexAtAtributeFromDataSet(dataSet.getHeaders(), dataSet.getTarget());
        if ( idxTarget == -1 ) {
            throw new Exception("Target not found");
        }
        for ( int i = 0; i < dataSet.getAttributeTypes().size(); i++ ) {
            if ( !dataSet.getAttributeTypes().get(i).equals(DataSet.NUMERIC_TYPE) ) {
                throw new Exception("The attribute " + dataSet.getHeaders().get(i) + " must be numeric");
            }
        }
        // Params
        if ( params == null ) {
            throw new Exception("Object params can not be null");
        }
        if ( !(params instanceof Backpropagation.Params) ) {
            throw new Exception("Params object is invalid");
        }
        if ( ((Params) params).epochs <= 0 ) {
            throw new Exception("Epochs cannot be 0 or less");
        }
        if ( ((Params) params).learningRate <= 0 || ((Params) params).learningRate > 1 ) {
            throw new Exception("The learning rate must be a value bounded between 0 and 1");
        }
        if ( ((Params) params).minError < 0 ) {
            throw new Exception("The minimum error cannot be less than 0");
        }
        // Gradiente
        boolean idxGradient = false;
        int tmpGradient = ((Params) params).gradient;
        for ( int gradient : GRADIENTS ) {
            if ( gradient == tmpGradient ) {
                idxGradient = true;
                break;
            }
        }
        if ( !idxGradient ) {
            throw new Exception("The specified gradient is invalid");
        }
        // Mini-batch
        if ( tmpGradient == MINI_BATCH_GRADIENT && ( ((Params) params).batchSize < 1 || ((Params) params).batchSize > dataSet.getInstances().size() ) ) {
            throw new Exception("The batch size is out of range (1 - " + dataSet.getInstances().size() + ") ");
        }
        // Capas
        ArrayList<Layer> layers = ((Params) params).hiddenLayers;
        if ( layers != null ) {
            for ( int i = 0; i < layers.size(); i++ ) {
                if ( layers.get(i).neurons < 0 ) {
                    throw new Exception("The number of neurons in hidden layer " + layers.get(i) + " cannot be less than 0");
                }
                boolean idxFunction = false;
                for ( int function : FUNCTIONS ) {
                    if ( function == layers.get(i).function ) {
                        idxFunction = true;
                        break;
                    }
                }
                if ( !idxFunction ) {
                    throw new Exception("The hidden layer " + layers.get(i) + 1 + " activation function is not valid");
                }
            }
        }
        // Funcion de la ultima capa
        boolean idxFunction = false;
        for ( int function : FUNCTIONS ) {
            if ( function == ((Params) params).lastNeuronFunctionActivation ) {
                idxFunction = true;
                break;
            }
        }
        if ( !idxFunction ) {
            throw new Exception("The output layer activation function is not valid");
        }
    }

    public static class Params extends AbstractParams {

        private double learningRate;
        private int epochs;
        private double minError;
        private ArrayList<Layer> hiddenLayers;
        private int gradient;
        private int batchSize;
        private int lastNeuronFunctionActivation;

        public void setLearningRate(double learningRate) {
            this.learningRate = learningRate;
        }

        public void setEpochs(int epochs) {
            this.epochs = epochs;
        }

        public void setMinError(double minError) {
            this.minError = minError;
        }

        public void setHiddenLayers(ArrayList<Layer> hiddenLayers) {
            this.hiddenLayers = hiddenLayers;
        }

        public void setGradient(int gradient) {
            this.gradient = gradient;
        }

        public void setBatchSize(int batchSize) {
            this.batchSize = batchSize;
        }

        public void setLastNeuronFunctionActivation(int lastNeuronFunctionActivation) {
            this.lastNeuronFunctionActivation = lastNeuronFunctionActivation;
        }

    }

    public static class Layer {

        private int neurons;
        private double[][] weights;
        private double[] outputs;
        private double[] sensivilities;
        private double[] netActivations;
        private boolean last;
        private int function;
        private double[][] gradient;

        public Layer(int neurons, int function)
        {
            this.neurons = neurons;
            this.function = function;
            last = false;
        }

        private Layer(int neurons, int function, boolean last)
        {
            this(neurons, function);
            this.last = last;
        }

        @Override
        public String toString() {
            return "Layer{" +
                    "neurons=" + neurons +
                    ", weights=" + Arrays.toString(weights) +
                    ", outputs=" + Arrays.toString(outputs) +
                    ", sensivilities=" + Arrays.toString(sensivilities) +
                    ", netActivations=" + Arrays.toString(netActivations) +
                    ", last=" + last +
                    ", function=" + function +
                    ", gradient=" + Arrays.toString(gradient) +
                    '}';
        }

    }

    public static class Model implements AbstractModel<Double> {

        private final DataSet dataSet;
        private final ArrayList<Layer> layers;
        private final HashMap<String, double[]> targets;
        private final double minError;
        private final double error;
        private final int epochs;
        private final int epoch;

        public Model(DataSet dataSet, ArrayList<Layer> layers, HashMap<String, double[]> targets, double minError, double error, int epochs, int epoch)
        {
            this.dataSet = dataSet;
            this.layers = layers;
            this.targets = targets;
            this.minError = minError;
            this.error = error;
            this.epochs = epochs;
            this.epoch = epoch;
        }

        public Object[] predictWithPercentage(Object[] instance) throws Exception {
            if ( instance.length != dataSet.getHeaders().size() - 1 ) {
                throw new Exception("The instance is not the same size as the data set (" + instance.length + " - " + (dataSet.getHeaders().size() - 1) + ")");
            }
            for ( Object value : instance ) {
                if ( !(value instanceof Double) ) {
                    throw new Exception("Some instance value is not a numerical value");
                }
            }
            double[] evaluate = new double[instance.length];
            for ( int i = 0; i < instance.length; i++ ) {
                evaluate[i] = (double) instance[i];
            }
            initFirstLayer(layers.get(0), evaluate);
            for ( int k = 1;  k < layers.size(); k++ ) {
                layers.get(k).netActivations = AlgebraicHelpers.matrixMultiplicationByVector(layers.get(k).weights, layers.get(k - 1).outputs);
                updateOutputsForForward(layers.get(k));
            }
            double[] output = layers.get(layers.size() - 1).outputs;
            int idxTarget = findIdxForTarget(output);
            String target = findTarget(output, idxTarget);
            return new Object[]{ target, output[idxTarget] };
        }

        private String findTarget(double[] output, int idx) {
            double[] newOutput = new double[output.length];
            Arrays.fill(newOutput, 0);
            newOutput[idx] = 1;
            String target = null;
            for ( Map.Entry<String, double[]> entry : targets.entrySet() ) {
                boolean isEquals = true;
                for ( int i = 0; i < newOutput.length; i++ ) {
                    if ( newOutput[i] != entry.getValue()[i] ) {
                        isEquals = false;
                        break;
                    }
                }
                if ( isEquals ) {
                    target = entry.getKey();
                    break;
                }
            }
            return target;
        }

        private int findIdxForTarget(double[] output) {
            int idx = -1;
            double maxValue = Double.MIN_VALUE;
            for ( int i = 0; i < output.length; i++ ) {
                if ( output[i] > maxValue ) {
                    idx = i;
                    maxValue = output[i];
                }
            }
            return idx;
        }

        @Override
        public Double predict(Object[] instance) throws Exception {
           throw new Exception("Not implemented yet");
        }

        @Override
        public void predict(DataSet dataSet, String classNameOut) throws Exception {
            throw new Exception("Not implemented yet");
        }

        @Override
        public String toString() {
            return "Model{" +
                    "layers=" + layers +
                    ", targets=" + targets +
                    ", minError=" + minError +
                    ", error=" + error +
                    ", epochs=" + epochs +
                    ", epoch=" + epoch +
                    '}';
        }
    }

}
