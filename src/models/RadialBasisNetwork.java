package models;

import utils.DataSet;
import utils.DataSetFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class RadialBasisNetwork extends Regression {

    public static final int KMEANS_RANDOM = 1;
    public static final int KMEANS_KFIRSTS = 2;
    public static final int KMEANS_KRAMDON_DATA = 3;

    private static final int[] INITIALIZATIONS = {
            KMEANS_RANDOM,
            KMEANS_KFIRSTS,
            KMEANS_KRAMDON_DATA
    };

    public static final int GAUSSIAN_FUNCTION = 1;
    public static final int INVERSE_MULTI_QUADRATIC_FUNCTION = 2;
    public static final int REFLECTED_SIGMOID = 3;

    private static final int[] RADIAL_BASIS_FUNCTIONS = {
            GAUSSIAN_FUNCTION,
            INVERSE_MULTI_QUADRATIC_FUNCTION,
            REFLECTED_SIGMOID
    };

    private RadialBasisNetwork()
    {

    }

    public static Model generateModel(DataSet dataSet, Params params) throws Exception {
        RadialBasisNetwork radialBasisNetwork = new RadialBasisNetwork();
        return radialBasisNetwork.makeModel(dataSet, params);
    }

    @Override
    protected Model makeModel(DataSet dataSet) throws Exception {
        throw new Exception("Not supported");
    }

    @Override
    protected Model makeModel(DataSet dataSet, AbstractParams params) throws Exception {
        validateDataSetAndParams(dataSet, params);
        int idxTarget = DataSetFunctions.getIndexAtAtributeFromDataSet(dataSet.getHeaders(), dataSet.getTarget());
        // Conjunto de datos
        double[] X = DataSetFunctions.generateNumericVectorAsDouble(dataSet, idxTarget == 0 ? 1 : 0);
        double[] Y = DataSetFunctions.generateNumericVectorAsDouble(dataSet, idxTarget);
        // Desempaquetado de parametros para kmeans
        int clusters = ((Params) params).clusters;
        int epochs = ((Params) params).epochs;
        int epochsForKmeans = 0;
        boolean changeInstances = true;
        // K-means
        double[] centroids = new double[clusters];
        initializeCentroids(centroids, X, ((Params) params).initializationKmeans);
        int[] newClasses = new int[X.length];
        Arrays.fill(newClasses, -1);
        // K-Means
        while ( changeInstances && epochsForKmeans < epochs ) {
            // Iteracion de las instancias
            changeInstances = false;
            for ( int i = 0; i < X.length; i++ ) {
                // Cálculo de una instancia contra los centroides
                double[] distances = new double[centroids.length];
                for ( int j = 0; j < centroids.length; j++ ) {
                    distances[j] = Math.sqrt(Math.pow(centroids[j] - X[i], 2));
                }
                // Obtenemos el que tiene menor distancia
                int idxMinDistance = -1;
                double mindDistance = Double.MAX_VALUE;
                for ( int j = 0; j < distances.length; j++ ) {
                    if ( distances[j] < mindDistance ) {
                        idxMinDistance = j;
                        mindDistance = distances[j];
                    }
                }
                if ( newClasses[i] != idxMinDistance ) {
                    changeInstances = true;
                }
                newClasses[i] = idxMinDistance;
            }
            // Actualizacion de los centroides
            for ( int i = 0; i < centroids.length; i++ ) {
                double newValue = 0.0;
                int values = 0;
                // Iteramos dentro de las instancias
                for ( int j = 0; j < X.length; j++ ) {
                    // Las que no pertenezcan no las tomamos
                    if ( newClasses[j] != i ) {
                        continue;
                    }
                    // Sumamos los valores
                    newValue += X[j];
                    values++;
                }
                // Promediamos los valores para obtener la posicion del centroide
                centroids[i] = newValue / values;
            }
            epochsForKmeans++;
        }
        // Radial Basis Network
        double[] weights = new double[centroids.length + 1];
        for ( int i = 0; i < weights.length; i++ ) {
            weights[i] = getRandom();
        }
        // Desempaquetado de parametros para kmeans
        int function = ((Params) params).radialBasisFunction;
        double minError = ((Params) params).minError;
        double error = Double.MAX_VALUE;
        int epochsForRadialBasis = 0;
        double learningRate = ((Params) params).learningRate;
        double sigma = getSigma(centroids, clusters);
        // Radial Basis Network
        while ( error > minError && epochsForRadialBasis < epochs ) {
            error = 0.0;
            for ( int i = 0; i < X.length; i++ ) {
                double[] outputs = getOutputsFromNeurons(centroids, sigma, X[i], function);
                double netValue = getNetValue(outputs, weights);
                double currentError = Y[i] - netValue;
                error += Math.pow( currentError, 2 );
                // Update weights
                for ( int j = 0; j < weights.length; j++ ) {
                    weights[j] = weights[j] + ( learningRate * currentError * outputs[j] );
                }
            }
            error /= X.length;
            epochsForRadialBasis++;
        }
        return new Model(dataSet, learningRate, epochs, epochsForKmeans, epochsForRadialBasis, minError, error, clusters, ((Params) params).initializationKmeans, function, centroids, weights, sigma);
    }

    private static double getNetValue(double[] outputs, double[] weights) {
        double result = 0;
        for ( int j = 0; j < outputs.length; j++ ) {
            result += weights[j] * outputs[j];
        }
        return result;
    }

    private static double[] getOutputsFromNeurons(double[] centroids, double sigma, double x, int function) {
        double[] outputs = new double[centroids.length + 1];
        if ( function == GAUSSIAN_FUNCTION ) {
            for ( int i = 0; i < centroids.length; i++ ) {
                double r = Math.pow(x - centroids[i], 2);
                outputs[i] = gaussianFunction(r, sigma);
            }
        } else if ( function == INVERSE_MULTI_QUADRATIC_FUNCTION ) {
            for ( int i = 0; i < centroids.length; i++ ) {
                double r = Math.pow(x - centroids[i], 2);
                outputs[i] = inverseMultiQuadraticFunction(r, sigma);
            }
        } else {
            for ( int i = 0; i < centroids.length; i++ ) {
                double r = Math.pow(x - centroids[i], 2);
                outputs[i] = reflectedSigmoidFunction(r, sigma);
            }
        }
        outputs[outputs.length - 1] = 1;
        return outputs;
    }

    private static double reflectedSigmoidFunction(double r, double sigma) {
        return 1 / ( 1 + Math.pow( Math.E, sigma * Math.pow(r, 2) ) );
    }

    private static double inverseMultiQuadraticFunction(double r, double sigma) {
        return Math.pow( Math.pow(r, 2) + Math.pow(sigma, 2), -0.5 );
    }

    private static double gaussianFunction(double r, double sigma) {
        return Math.pow(Math.E, - (Math.pow(r, 2) / (2 * Math.pow(sigma, 2)) ) );
    }

    private double getSigma(double[] centroids, int clusters) {
        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        for ( int i = 0; i < centroids.length; i++ ) {
            max = Math.max(centroids[i], max);
            min = Math.min(centroids[i], min);
        }
        return (max - min) / ( Math.sqrt(2 * clusters) );
    }

    private double getRandom() {
        int random = (int) (Math.random() * (1000));
        int sign = (int) (Math.random() * 10);
        return (double) ((sign % 2 == 0) ? -random : random) / 1000;
    }

    private void initializeCentroids(double[] centroids, double[] X, int initializer) {
        if ( initializer == KMEANS_RANDOM ) {
            double max = Double.MIN_VALUE;
            double min = Double.MAX_VALUE;
            for ( int i = 0; i < X.length; i++ ) {
                if ( X[i] < min ) {
                    min = X[i];
                }
                if ( X[i] > max ) {
                    max = X[i];
                }
            }
            for ( int i = 0; i < centroids.length; i++ ) {
                centroids[i] = ( Math.random() * ( max - Math.abs(min) )) + min;
            }
            return;
        }
        if ( initializer == KMEANS_KFIRSTS ) {
            System.arraycopy(X, 0, centroids, 0, centroids.length);
            return;
        }
        // KMENAS_KRAMDON_DATA
        ArrayList<Integer> idxs = new ArrayList<>();
        for ( int i = 0; i < X.length; i++ ) {
            idxs.add(i);
        }
        Collections.shuffle(idxs);
        for ( int i = 0; i < centroids.length; i++ ) {
            centroids[i] = X[idxs.get(i)];
        }
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
        if ( dataSet.getHeaders().size() != 2 ) {
            throw new Exception("The dataset can only have 2 attributes");
        }
        // Params
        if ( params == null ) {
            throw new Exception("Object params can not be null");
        }
        if ( !(params instanceof RadialBasisNetwork.Params) ) {
            throw new Exception("Params object is invalid");
        }
        if ( ((Params) params).epochs <= 0 ) {
            throw new Exception("Epochs cannot be 0 or less");
        }
        if ( ((Params) params).learningRate <= 0 || ((RadialBasisNetwork.Params) params).learningRate > 1 ) {
            throw new Exception("The learning rate must be a value bounded between 0 and 1");
        }
        if ( ((Params) params).minError < 0 ) {
            throw new Exception("The minimum error cannot be less than 0");
        }
        if ( ((Params) params).clusters <= 0 || ((Params) params).clusters > dataSet.getInstances().size() ) {
            throw new Exception("The number of clusters cannot be greater than the number of instances or zero");
        }
        // Inicializacion
        boolean idxInitialization = false;
        int tmpInitialization = ((Params) params).initializationKmeans;
        for ( int intialization : INITIALIZATIONS ) {
            if ( intialization == tmpInitialization ) {
                idxInitialization = true;
                break;
            }
        }
        if ( !idxInitialization ) {
            throw new Exception("The specified initialization of k-means is invalid");
        }
        // Function
        boolean idxFunction = false;
        int tmpFunction = ((Params) params).radialBasisFunction;
        for ( int function : RADIAL_BASIS_FUNCTIONS ) {
            if ( tmpFunction == function ) {
                idxFunction = true;
                break;
            }
        }
        if ( !idxFunction ) {
            throw new Exception("The specified radial basis function is invalid");
        }
    }

    public static class Params extends AbstractParams {

        private double learningRate;
        private int epochs;
        private double minError;
        private int clusters;
        private int initializationKmeans;
        private int radialBasisFunction;

        public void setLearningRate(double learningRate) {
            this.learningRate = learningRate;
        }

        public void setEpochs(int epochs) {
            this.epochs = epochs;
        }

        public void setMinError(double minError) {
            this.minError = minError;
        }

        public void setClusters(int clusters) {
            this.clusters = clusters;
        }

        public void setInitializationKmeans(int initializationKmeans) {
            this.initializationKmeans = initializationKmeans;
        }

        public void setRadialBasisFunction(int radialBasisFunction) {
            this.radialBasisFunction = radialBasisFunction;
        }

    }

    public static class Model implements AbstractModel<Double> {

        private final DataSet dataSet;
        private final double learningRate;
        private final int epochs;
        private final int epochsForKmeans;
        private final int epochsForRadialBasis;
        private final double minError;
        private final double error;
        private final int clusters;
        private final int initializationKmeans;
        private final int radialBasisFunction;
        private final double[] centroids;
        private final double[] weights;
        private final double sigma;

        public Model(DataSet dataSet, double learningRate, int epochs, int epochsForKmeans, int epochsForRadialBasis, double minError, double error, int clusters, int initializationKmeans, int radialBasisFunction, double[] centroids, double[] weights, double sigma) {
            this.dataSet = dataSet;
            this.learningRate = learningRate;
            this.epochs = epochs;
            this.epochsForKmeans = epochsForKmeans;
            this.epochsForRadialBasis = epochsForRadialBasis;
            this.minError = minError;
            this.error = error;
            this.clusters = clusters;
            this.initializationKmeans = initializationKmeans;
            this.radialBasisFunction = radialBasisFunction;
            this.centroids = centroids;
            this.weights = weights;
            this.sigma = sigma;
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
            double newInstance = (double) instance[0];
            double[] outputs = getOutputsFromNeurons(centroids, sigma, newInstance, radialBasisFunction);
            return getNetValue(outputs, weights);
        }

        @Override
        public void predict(DataSet dataSet, String classNameOut) throws Exception {
            throw new Exception("Not implemented yet");
        }

        @Override
        public String toString() {
            return "Model{" +
                    "learningRate=" + learningRate +
                    ", epochs=" + epochs +
                    ", epochsForKmeans=" + epochsForKmeans +
                    ", epochsForRadialBasis=" + epochsForRadialBasis +
                    ", minError=" + minError +
                    ", error=" + error +
                    ", clusters=" + clusters +
                    ", initializationKmeans=" + initializationKmeans +
                    ", radialBasisFunction=" + radialBasisFunction +
                    ", \ncentroids=" + Arrays.toString(centroids) +
                    ", \nweights=" + Arrays.toString(weights) +
                    ", \nsigma=" + sigma +
                    '}';
        }
    }

}
