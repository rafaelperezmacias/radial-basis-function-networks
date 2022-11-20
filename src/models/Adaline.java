package models;

import utils.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Adaline extends Regression {

    public static final int SIGMOID_FUNCTION = 1;
    public static final int ANOTHER_FUNCTION = 2;

    private Adaline()
    {

    }

    public static Model generateModel(DataSet dataSet, Params params) throws Exception {
        Adaline perceptron = new Adaline();
        return perceptron.makeModel(dataSet, params);
    }

    @Override
    protected Model makeModel(DataSet dataSet) throws Exception {
        throw new Exception("Not supported");
    }

    @Override
    protected Model makeModel(DataSet dataSet, AbstractParams params) throws Exception {
        validateDataSetAndParams(dataSet, params);
        int idxTarget = DataSetFunctions.getIndexAtAtributeFromDataSet(dataSet.getHeaders(), dataSet.getTarget());

        double[][] X = generateAnXMatrix(DataSetFunctions.generateNumericMatrix(dataSet, getColumns(dataSet, idxTarget)));
        double[] Y = DataSetFunctions.generateNumericVectorAsDouble(dataSet, idxTarget);

        double[] weights = ((Params) params).weights;
        double learningRate = ((Params) params).learningRate;
        double minError = ((Params) params).minError;
        double error = Double.MAX_VALUE;
        int epochs = ((Params) params).epochs;
        int epoch = 0;
        int function = ((Params) params).function;

        while ( error > minError && epoch < epochs ) {
            error = 0.0;
            for ( int i = 0; i < X.length; i++ ) {
                double activationValue = AlgebraicHelpers.getProductPoint(weights, X[i]);
                double currentError = Y[i] - getFunctionActivation(activationValue, function);
                error += Math.pow( currentError, 2 );
                for ( int j = 0; j < weights.length; j++ ) {
                    weights[j] = weights[j] + ( learningRate * currentError * getDerivativeFunction(activationValue, function) * X[i][j] );
                }
            }
            epoch++;
        }

        return new Model(dataSet, epochs, epoch, weights, learningRate, minError, error, function);
    }

    private static double getFunctionActivation(double activationValue, int function) {
        if ( function == SIGMOID_FUNCTION ) {
            return Functions.sigmoid(activationValue);
        }
        return 0.0;
    }

    private static double getDerivativeFunction(double activationValue, int function) {
        if ( function == SIGMOID_FUNCTION ) {
            return Functions.derivativeSigmoid(activationValue);
        }
        return 0.0;
    }

    private static double[][] generateAnXMatrix(double[][] matrix) {
        double[][] newMatrix = new double[matrix.length][matrix[0].length + 1];
        for ( int i = 0; i < matrix.length; i++ ) {
            newMatrix[i][0] = -1;
        }
        for ( int i = 1; i < newMatrix[0].length; i++ ) {
            for ( int j = 0; j < newMatrix.length; j++ ) {
                newMatrix[j][i] = matrix[j][i - 1];
            }
        }
        return newMatrix;
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
        if ( !(params instanceof Adaline.Params) ) {
            throw new Exception("Params object is invalid");
        }
        if ( ((Params) params).weights == null ) {
            throw new Exception("The weights of the parameters can not be null");
        }
        if ( ((Params) params).weights.length != dataSet.getHeaders().size() ) {
            throw new Exception("The weights indicated do not match those of the data set");
        }
        // Target
        Set<String> targetsFromDataSet = new HashSet<>();
        for ( int i = 0; i < dataSet.getInstances().size(); i++ ) {
            targetsFromDataSet.add( dataSet.getInstances().get(i).get(idxTarget) );
        }
        for ( String target : targetsFromDataSet ) {
            if ( !target.equals("1") && !target.equals("0") ) {
                throw new Exception("Class values are not allowed, only 0 and 1 are allowed");
            }
        }
    }

    public static class Params extends AbstractParams {

        private double learningRate;
        private int epochs;
        private double minError;
        private double[] weights;
        private int function;

        public void setLearningRate(double learningRate) {
            this.learningRate = learningRate;
        }

        public void setEpochs(int epochs) {
            this.epochs = epochs;
        }

        public void setWeights(double[] weights) {
            this.weights = weights;
        }

        public void setMinError(double minError) {
            this.minError = minError;
        }

        public void setFunction(int function) {
            this.function = function;
        }

    }

    public static class Model implements AbstractModel<Double> {

        private final DataSet dataSet;
        private final int epochs;
        private final int epoch;
        private final double[] weights;
        private final double learningRate;
        private final double minError;
        private final double error;
        private final int function;

        public Model(DataSet dataSet, int epochs, int epoch, double[] weights, double learningRate, double minError, double error, int function)
        {
            this.dataSet = dataSet;
            this.epochs = epochs;
            this.epoch = epoch;
            this.weights = weights;
            this.learningRate = learningRate;
            this.minError = minError;
            this.error = error;
            this.function = function;
        }

        @Override
        public Double predict(Object[] instance) throws Exception {
            if ( instance.length != dataSet.getHeaders().size() - 1 ) {
                throw new Exception("The instance is not the same size as the data set (" + instance.length + " - " + (dataSet.getHeaders().size() - 1) + ")");
            }
            for ( Object value : instance ) {
                if ( !(value instanceof Double) ) {
                    throw new Exception("Some instance value is not a numerical value");
                }
            }
            double[] newInstance = new double[instance.length];
            for ( int i = 0; i < instance.length; i++ ) {
                newInstance[i] = (double) instance[i];
            }
            double[] evaluate = new double[instance.length + 1];
            evaluate[0] = -1;
            System.arraycopy(newInstance, 0, evaluate, 1, instance.length);
            double activationValue = AlgebraicHelpers.getProductPoint(weights, evaluate);
            return getFunctionActivation(activationValue, function);
        }

        @Override
        public void predict(DataSet dataSet, String classNameOut) throws Exception {
            throw new Exception("Not implemented yet");
        }

        @Override
        public String toString() {
            return "Model{" +
                    "epochs=" + epochs +
                    ", epoch=" + epoch +
                    ", weights=" + Arrays.toString(weights) +
                    ", learningRate=" + learningRate +
                    ", minError=" + minError +
                    ", error=" + error +
                    ", function=" + function +
                    '}';
        }
    }

}
