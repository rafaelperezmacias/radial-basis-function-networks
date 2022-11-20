package windows;

import models.AbstractModel;
import models.AbstractParams;
import models.RadialBasisNetwork;
import models.Regression;
import utils.DataSet;
import utils.DataSetFunctions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class RadialBasisNetworkThread extends Regression implements Runnable {

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

    private DataSet dataSet;
    private AbstractParams params;
    private RBFWindow rbfWindow;

    public RadialBasisNetworkThread()
    {

    }

    public void generateModel(DataSet dataSet, Params params, RBFWindow rbfWindow) {
        this.dataSet = dataSet;
        this.params = params;
        this.rbfWindow = rbfWindow;
        Thread thread = new Thread(this);
        thread.start();
    }

    @Override
    protected Model makeModel(DataSet dataSet) throws Exception {
        throw new Exception("Not supported");
    }

    @Override
    protected Model makeModel(DataSet dataSet, AbstractParams params) throws Exception {
        throw new Exception("Not supported");
    }

    @Override
    public void run() {
        try {
            validateDataSetAndParams(dataSet, params);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        int idxTarget = DataSetFunctions.getIndexAtAtributeFromDataSet(dataSet.getHeaders(), dataSet.getTarget());
        // Conjunto de datos
        double[] X = new double[0];
        try {
            X = DataSetFunctions.generateNumericVectorAsDouble(dataSet, idxTarget == 0 ? 1 : 0);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        double[] Y = new double[0];
        try {
            Y = DataSetFunctions.generateNumericVectorAsDouble(dataSet, idxTarget);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
        rbfWindow.printCentroids(centroids,false);
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
            rbfWindow.printCentroids(centroids, false);
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
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
        double sigma = 0;
        try {
            sigma = getSigma(centroids, clusters);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
            rbfWindow.updateValuesForUI(error, epochsForRadialBasis, false, error < minError);
            if ( epochsForRadialBasis < 500 ) {
                double[] yCentroids = getYCentroids(centroids, weights, sigma, function);
                Model model = new Model(dataSet, learningRate, epochs, epochsForKmeans, epochsForRadialBasis, minError, error, clusters, ((Params) params).initializationKmeans, function, centroids, weights, sigma);
                rbfWindow.setModel(model, false);
                rbfWindow.printCentroids(centroids, yCentroids, true);
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            } else if ( epochsForRadialBasis % 20 == 0 ) {
                double[] yCentroids = getYCentroids(centroids, weights, sigma, function);
                Model model = new Model(dataSet, learningRate, epochs, epochsForKmeans, epochsForRadialBasis, minError, error, clusters, ((Params) params).initializationKmeans, function, centroids, weights, sigma);
                rbfWindow.setModel(model, false);
                rbfWindow.printCentroids(centroids, yCentroids, true);
                try {
                    Thread.sleep(20);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            epochsForRadialBasis++;
        }
        rbfWindow.updateValuesForUI(error, epochsForRadialBasis, true, epochsForRadialBasis <= epochs && error < minError);
        Model model = new Model(dataSet, learningRate, epochs, epochsForKmeans, epochsForRadialBasis, minError, error, clusters, ((Params) params).initializationKmeans, function, centroids, weights, sigma);
        double[] yCentroids = getYCentroids(centroids, weights, sigma, function);
        rbfWindow.printCentroids(centroids, yCentroids, true);
        rbfWindow.setModel(model, true);
    }

    private static double[] getYCentroids(double[] centroids, double[] weights, double sigma, int function) {
        double[] result = new double[centroids.length];
        for ( int i = 0; i < centroids.length; i++ ) {
            double[] outputs = getOutputsFromNeurons(centroids, sigma, centroids[i], function);
            result[i] = getNetValue(outputs, weights);
        }
        return result;
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
        return Math.pow( Math.pow(r, 2) + Math.pow(sigma, 2), -(double) 1/2 );
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
                centroids[i] = ( Math.random() * ( max + Math.abs(min) )) + min;
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
        if ( !(params instanceof RadialBasisNetworkThread.Params) ) {
            throw new Exception("Params object is invalid");
        }
        if ( ((RadialBasisNetworkThread.Params) params).epochs <= 0 ) {
            throw new Exception("Epochs cannot be 0 or less");
        }
        if ( ((RadialBasisNetworkThread.Params) params).learningRate <= 0 || ((RadialBasisNetworkThread.Params) params).learningRate > 1 ) {
            throw new Exception("The learning rate must be a value bounded between 0 and 1");
        }
        if ( ((RadialBasisNetworkThread.Params) params).minError < 0 ) {
            throw new Exception("The minimum error cannot be less than 0");
        }
        if ( ((RadialBasisNetworkThread.Params) params).clusters <= 0 || ((RadialBasisNetworkThread.Params) params).clusters > dataSet.getInstances().size() ) {
            throw new Exception("The number of clusters cannot be greater than the number of instances or zero");
        }
        // Inicializacion
        boolean idxInitialization = false;
        int tmpInitialization = ((RadialBasisNetworkThread.Params) params).initializationKmeans;
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
        int tmpFunction = ((RadialBasisNetworkThread.Params) params).radialBasisFunction;
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
