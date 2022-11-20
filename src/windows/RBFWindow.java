package windows;

import models.RadialBasisNetwork;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RectangleInsets;
import org.jfree.util.ShapeUtilities;
import utils.DataSet;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Arrays;

public class RBFWindow extends JFrame {

    private ArrayList<Point> points;

    private final int MAP_WIDTH = 400;
    private final int MAP_HEIGHT = 400;
    private final int RADIUS_POINT = 5;
    private final int RADIUS_POINT_CENTROID = 4;
    private final double MAP_SCALE = 5.0;

    private final Map map;

    private final JMenu jmOptions;
    private final JMenu jmKmeans;
    private final JMenu jmRBF;

    private final JTextField txtLearningRate;
    private final JTextField txtEpochs;
    private final JTextField txtMinError;
    private final JTextField txtClusters;

    private XYSeries errorSeries;

    private final JButton btnRBF;
    private final JButton btnForm1;
    private final JButton btnForm2;
    private final JButton btnForm3;

    private final JLabel lblEpochResult;
    private final JLabel lblErrorResult;

    private final String unicodeSuperindexTwo = "\u00B2";

    private RadialBasisNetworkThread.Model model;

    private int radialBasisFunction;
    private int initializationKMeans;
    private boolean clickable;

    public RBFWindow()
    {
        super("Radial Basis Function Network");
        setLayout(null);
        setSize(1012,730);
        setLocationRelativeTo(null);
        // Inicializamos la lista que contiene los puntos del mapa
        points = new ArrayList<>();
        // Barra de menu
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        // Un menu de la barra
        jmOptions = new JMenu("Opciones");
        menuBar.add(jmOptions);
        // Opciones del menu
        // ELiminar ultima
        JMenuItem jmiDeleteLastInstance = new JMenuItem("Eliminar la ultima instancia");
        jmiDeleteLastInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !points.isEmpty() ) {
                    int idxPoint = -1;
                    for ( int i = points.size() - 1; i >= 0; i-- ) {
                        if ( points.get(i).type == 0 ) {
                            idxPoint = i;
                            break;
                        }
                    }
                    if ( idxPoint != -1 ) {
                        points.remove(idxPoint);
                        map.repaint();
                    }
                }
            }
        });
        jmOptions.add(jmiDeleteLastInstance);
        // Limpiar instancias
        JMenuItem jmiClearInstances = new JMenuItem("Limpiar instancias");
        jmiClearInstances.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                map.repaint();
                btnForm1.setEnabled(true);
                btnForm2.setEnabled(true);
                btnForm3.setEnabled(true);
            }
        });
        jmOptions.add(jmiClearInstances);
        // Limpiar el programa
        JMenuItem jmiClearAll = new JMenuItem("Limpiar todo");
        jmiClearAll.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                txtLearningRate.setText("0.");
                txtEpochs.setText("");
                txtMinError.setText("");
                txtClusters.setText("");
                btnForm1.setEnabled(true);
                btnForm2.setEnabled(true);
                btnForm3.setEnabled(true);
                points.clear();
                map.repaint();
                errorSeries.clear();
                lblEpochResult.setText("<html>Epoca: <b>0</b></html>");
                lblErrorResult.setText("<html>Error: <b>0.0</b></html>");
            }
        });
        jmOptions.add(jmiClearAll);
        // Salir
        JMenuItem jmiClose = new JMenuItem("Salir");
        jmiClose.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                RBFWindow.this.dispose();
            }
        });
        jmOptions.add(jmiClose);
        // Lienzo princiapal de la ventana
        map = new Map();
        map.setSize(MAP_WIDTH, MAP_HEIGHT);
        map.setLocation(35,30);
        map.setBackground(Color.WHITE);
        add(map);
        // Menu de K-means
        jmKmeans = new JMenu("K-means");
        ButtonGroup bgKmeams = new ButtonGroup();
        menuBar.add(jmKmeans);
        JRadioButtonMenuItem jmiKRandom = new JRadioButtonMenuItem("Aleatorio");
        jmiKRandom.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                initializationKMeans = RadialBasisNetworkThread.KMEANS_RANDOM;
            }
        });
        bgKmeams.add(jmiKRandom);
        jmKmeans.add(jmiKRandom);
        JRadioButtonMenuItem jmiKFirst = new JRadioButtonMenuItem("K primeros");
        jmiKFirst.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                initializationKMeans = RadialBasisNetworkThread.KMEANS_KFIRSTS;
            }
        });
        jmiKFirst.setSelected(true);
        bgKmeams.add(jmiKFirst);
        jmKmeans.add(jmiKFirst);
        JRadioButtonMenuItem jmiKRandomK = new JRadioButtonMenuItem("K aleatorios");
        jmiKRandomK.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                initializationKMeans = RadialBasisNetworkThread.KMEANS_KRAMDON_DATA;
            }
        });
        bgKmeams.add(jmiKRandomK);
        jmKmeans.add(jmiKRandomK);
        // Menu de RBF
        jmRBF = new JMenu("Radial Basis Function");
        ButtonGroup bgFunctions = new ButtonGroup();
        menuBar.add(jmRBF);
        JRadioButtonMenuItem jmiGaussian = new JRadioButtonMenuItem("Gaussiana");
        jmiGaussian.setSelected(true);
        jmiGaussian.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                radialBasisFunction = RadialBasisNetworkThread.GAUSSIAN_FUNCTION;
            }
        });
        jmRBF.add(jmiGaussian);
        bgFunctions.add(jmiGaussian);
        JRadioButtonMenuItem jmiIMQF = new JRadioButtonMenuItem("Inversa multi-cuadratica");
        jmiIMQF.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                radialBasisFunction = RadialBasisNetworkThread.INVERSE_MULTI_QUADRATIC_FUNCTION;
            }
        });
        jmRBF.add(jmiIMQF);
        bgFunctions.add(jmiIMQF);
        JRadioButtonMenuItem jmiReflectedSigmoid = new JRadioButtonMenuItem("Sigmoide reflejada");
        jmiReflectedSigmoid.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                radialBasisFunction = RadialBasisNetworkThread.REFLECTED_SIGMOID;
            }
        });
        jmRBF.add(jmiReflectedSigmoid);
        bgFunctions.add(jmiReflectedSigmoid);
        // Variables
        initializationKMeans = RadialBasisNetworkThread.KMEANS_KFIRSTS;
        radialBasisFunction = RadialBasisNetworkThread.GAUSSIAN_FUNCTION;
        clickable = true;
        // Eventos del mouse
        map.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point newPoint = new Point();
                // Tratamiento de los datos
                newPoint.xMap = e.getX();
                newPoint.yMap = e.getY();
                newPoint.x = ( e.getX() >= MAP_WIDTH * 0.5 ) ? e.getX() - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - e.getX());
                newPoint.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                newPoint.y = ( e.getY() > MAP_HEIGHT * 0.5 ) ? -(e.getY() - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - e.getY();
                newPoint.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                // Boton izquierdo
                if ( e.getButton() == MouseEvent.BUTTON1 && clickable ) {
                    ArrayList<Point> newPoints = new ArrayList<>();
                    for ( Point point : points ) {
                        if ( point.type == 0 ) {
                            newPoints.add(point);
                        }
                    }
                    btnForm1.setEnabled(true);
                    btnForm2.setEnabled(true);
                    btnForm3.setEnabled(true);
                    points = newPoints;
                    newPoint.type = 0;
                    newPoint.color = Color.BLACK;
                    points.add(newPoint);
                    System.out.println("Nuevo punto agregado: " + newPoint);
                    map.repaint();
                }
            }
        });
        /** Titulos, leyendas */
        // Escala del norte del plano
        JLabel lblScaleNorth = new JLabel("+ " + MAP_SCALE);
        lblScaleNorth.setSize(28,10);
        lblScaleNorth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() - 15);
        add(lblScaleNorth);
        // Escala del sur del plano
        JLabel lblScaleSouth = new JLabel("- " + MAP_SCALE);
        lblScaleSouth.setSize(28,10);
        lblScaleSouth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() + map.getHeight() + 5);
        add(lblScaleSouth);
        // Escala del este del plano
        JLabel lblScaleEast = new JLabel("+ " + MAP_SCALE);
        lblScaleEast.setSize(28,10);
        lblScaleEast.setLocation(map.getX() + ( map.getWidth() ) + 5, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleEast);
        // Escala del este del plano
        JLabel lblScaleWest = new JLabel("- " + MAP_SCALE);
        lblScaleWest.setSize(28,10);
        lblScaleWest.setLocation(map.getX() - 27, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleWest);
        // Configuracion de las instancias
        JLabel lblSettings = new JLabel("Configuracion");
        lblSettings.setLocation(0, map.getY() + map.getHeight() + 10);
        lblSettings.setSize(getWidth(), 24);
        lblSettings.setHorizontalAlignment(JLabel.CENTER);
        lblSettings.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblSettings);
        // Factor de aprendizaje
        JLabel lblLearningRate = new JLabel("Learning rate: ");
        lblLearningRate.setLocation(lblSettings.getX() + (int) ((lblSettings.getWidth() * 0.50) * 0.05), lblSettings.getY() + lblSettings.getHeight() + 10);
        lblLearningRate.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.20), 30);
        lblLearningRate.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblLearningRate);
        txtLearningRate = new JTextField("0.");
        txtLearningRate.setLocation(lblLearningRate.getX() + lblLearningRate.getWidth() + ((int) ((lblSettings.getWidth() * 0.50) * .02)), lblLearningRate.getY());
        txtLearningRate.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.63), 30);
        txtLearningRate.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' || txtLearningRate.getCaretPosition() < 2 ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
            @Override
            public void keyPressed(KeyEvent e) {
                if ( txtLearningRate.getText().length() == 2 && e.getKeyCode() == KeyEvent.VK_BACK_SPACE ) {
                    e.consume();
                }
                super.keyPressed(e);
            }
        });
        add(txtLearningRate);
        // Epocas
        JLabel lblEpochs = new JLabel("Epocas: ");
        lblEpochs.setSize(lblLearningRate.getSize());
        lblEpochs.setLocation(lblLearningRate.getX(), lblLearningRate.getY() + lblLearningRate.getHeight() + 5);
        lblEpochs.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblEpochs);
        txtEpochs = new JTextField();
        txtEpochs.setSize(txtLearningRate.getSize());
        txtEpochs.setLocation(txtLearningRate.getX(), lblEpochs.getY());
        txtEpochs.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtEpochs);
        // Error minimo
        JLabel lblError = new JLabel("Error minimo: ");
        lblError.setLocation(lblEpochs.getX(), lblEpochs.getY() + lblEpochs.getHeight() + 5);
        lblError.setSize(lblLearningRate.getSize());
        lblError.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblError);
        txtMinError = new JTextField();
        txtMinError.setLocation(txtLearningRate.getX(), lblError.getY());
        txtMinError.setSize(txtLearningRate.getSize());
        txtMinError.addKeyListener(new CustomKeyListener(txtMinError));
        add(txtMinError);
        // Clusters
        JLabel lblClusters = new JLabel("Clusters: ");
        lblClusters.setLocation(lblEpochs.getX(), lblError.getY() + lblError.getHeight() + 5);
        lblClusters.setSize(lblLearningRate.getSize());
        lblClusters.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblClusters);
        txtClusters = new JTextField();
        txtClusters.setLocation(txtLearningRate.getX(), lblClusters.getY());
        txtClusters.setSize(txtLearningRate.getSize());
        txtClusters.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtClusters);
        // Separador
        JSeparator jsSettings = new JSeparator();
        jsSettings.setOrientation(SwingConstants.VERTICAL);
        jsSettings.setSize(2, 150);
        jsSettings.setLocation((int) (lblSettings.getWidth() * 0.50), lblSettings.getY() + lblSettings.getHeight() + 5);
        add(jsSettings);
        // Entrenar
        btnRBF = new JButton("Radial Basis Function Network");
        btnRBF.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.80), 35);
        btnRBF.setLocation((int) (((lblSettings.getWidth() * 0.50) * 0.10) + (lblSettings.getWidth() * 0.50)), lblLearningRate.getY());
        btnRBF.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnRBF.setBackground(new Color(71, 138, 201));
        btnRBF.setOpaque(true);
        btnRBF.setFont(new Font("Dialog", Font.BOLD, 14));
        btnRBF.setForeground(Color.WHITE);
        btnRBF.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int epochs;
                double minError;
                double learningRate;
                int clusters;
                try {
                    learningRate = Double.parseDouble(txtLearningRate.getText());
                    epochs = Integer.parseInt(txtEpochs.getText());
                    minError = Double.parseDouble(txtMinError.getText());
                    clusters = Integer.parseInt(txtClusters.getText());
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(null, "Parametros no especificados o incorrectos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( epochs <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las epocas no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( learningRate == 0 ) {
                    JOptionPane.showMessageDialog(null, "El factor de aprendizaje no puede ser 0", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( minError < 0 ) {
                    JOptionPane.showMessageDialog(null, "El error no puede ser negativo", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }

                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point tmpPoint : points ) {
                    if ( tmpPoint.type == 0 || tmpPoint.type == 2 ) {
                        newPoints.add(tmpPoint);
                    }
                }
                points = newPoints;
                map.repaint();
                if ( clusters <= 0 || clusters > points.size() ) {
                    JOptionPane.showMessageDialog(null, "Clusters fuera de rango (1 - " + points.size() + ")", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                errorSeries.clear();
                lblEpochResult.setText("<html>Epoca: <b>0</b></html>");
                lblErrorResult.setText("<html>Error: <b>0.0</b></html>");
                changeUIState(false);
                clickable = false;
                // Creacion del conjunto de datos
                String[] headers = { "x", "y" };
                String[] attributeTypes = { DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE };
                DataSet dataSet;
                try {
                    dataSet = DataSet.getEmptyDataSetWithHeaders(headers, attributeTypes, "y");
                } catch (Exception ex) {
                    System.out.println("El dataset no pudo ser creado");
                    return;
                }
                for ( Point point : points ) {
                    try {
                        dataSet.addInstance(new ArrayList<>(Arrays.asList("" + point.x,"" + point.y)));
                    } catch (Exception ex) {
                        System.out.println("No se pudo agregar la instancia del punto " + point);
                    }
                }
                System.out.println("Conjunto de datos con el que el algoritmo trabajara");
                System.out.println(dataSet);
                // Parametros del algoritmo
                RadialBasisNetworkThread.Params params = new RadialBasisNetworkThread.Params();
                params.setLearningRate(learningRate);
                params.setEpochs(epochs);
                params.setMinError(minError);
                params.setClusters(clusters);
                params.setInitializationKmeans(initializationKMeans);
                params.setRadialBasisFunction(radialBasisFunction);
                try {
                    RadialBasisNetworkThread radialBasisNetworkThread = new RadialBasisNetworkThread();
                    radialBasisNetworkThread.generateModel(dataSet, params, RBFWindow.this);
                } catch ( Exception ex ) {
                    changeUIState(true);
                    clickable = true;
                    System.out.println("El modelo no se pudo generar");
                    ex.printStackTrace();
                }
            }
        });
        add(btnRBF);
        // Primera ecuacion
        btnForm1 = new JButton("((x-2)(2x+1))/(1+x"+ unicodeSuperindexTwo +")");
        btnForm1.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.60), 35);
        btnForm1.setLocation((int) (((lblSettings.getWidth() * 0.50) * 0.20) + (lblSettings.getWidth() * 0.50)), btnRBF.getY() + btnRBF.getHeight() + 5);
        btnForm1.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnForm1.setFont(new Font("Dialog", Font.PLAIN, 16));
        btnForm1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                for ( int i = 0; i < map.getWidth(); i++ ) {
                    Point point = new Point();
                    point.xMap = i;
                    point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                    point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                    double y = ((point.x - 2)*((2 * point.x) + 1))/(1 + Math.pow(point.x, 2));
                    point.y = y;
                    point.yMap = ( y >= 0 ) ? (int) ((MAP_HEIGHT * 0.5) - (y * ((MAP_HEIGHT * 0.5) / MAP_SCALE))) : (int) ((MAP_HEIGHT * 0.5) + (y * -((MAP_HEIGHT * 0.5) / MAP_SCALE)));
                    point.color = Color.BLACK;
                    point.type = 2;
                    points.add(point);
                }
                map.repaint();
                btnForm1.setEnabled(false);
                btnForm2.setEnabled(true);
                btnForm3.setEnabled(true);
                btnRBF.requestFocus();
            }
        });
        add(btnForm1);
        // Segunda ecuacion
        btnForm2 = new JButton("sin(x)");
        btnForm2.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.60), 35);
        btnForm2.setLocation((int) (((lblSettings.getWidth() * 0.50) * 0.20) + (lblSettings.getWidth() * 0.50)), btnForm1.getY() + btnForm1.getHeight());
        btnForm2.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnForm2.setFont(new Font("Dialog", Font.PLAIN, 16));
        btnForm2.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                for ( int i = 0; i < map.getWidth(); i++ ) {
                    Point point = new Point();
                    point.xMap = i;
                    point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                    point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                    double y = Math.sin(point.x);
                    point.y = y;
                    point.yMap = ( y >= 0 ) ? (int) ((MAP_HEIGHT * 0.5) - (y * ((MAP_HEIGHT * 0.5) / MAP_SCALE))) : (int) ((MAP_HEIGHT * 0.5) + (y * -((MAP_HEIGHT * 0.5) / MAP_SCALE)));
                    point.color = Color.BLACK;
                    point.type = 2;
                    points.add(point);
                }
                btnForm1.setEnabled(true);
                btnForm2.setEnabled(false);
                btnForm3.setEnabled(true);
                btnRBF.requestFocus();
                map.repaint();
            }
        });
        add(btnForm2);
        // Tercera ecuacion
        btnForm3 = new JButton("2sin(x) + cos(3x)");
        btnForm3.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.60), 35);
        btnForm3.setLocation((int) (((lblSettings.getWidth() * 0.50) * 0.20) + (lblSettings.getWidth() * 0.50)), btnForm2.getY() + btnForm2.getHeight());
        btnForm3.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnForm3.setFont(new Font("Dialog", Font.PLAIN, 16));
        btnForm3.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                for ( int i = 0; i < map.getWidth(); i++ ) {
                    Point point = new Point();
                    point.xMap = i;
                    point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                    point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                    double y = 2 * Math.sin(point.x) + Math.cos(point.x * 3);
                    point.y = y;
                    point.yMap = ( y >= 0 ) ? (int) ((MAP_HEIGHT * 0.5) - (y * ((MAP_HEIGHT * 0.5) / MAP_SCALE))) : (int) ((MAP_HEIGHT * 0.5) + (y * -((MAP_HEIGHT * 0.5) / MAP_SCALE)));
                    point.color = Color.BLACK;
                    point.type = 2;
                    points.add(point);
                }
                btnForm1.setEnabled(true);
                btnForm2.setEnabled(true);
                btnForm3.setEnabled(false);
                btnRBF.requestFocus();
                map.repaint();
            }
        });
        add(btnForm3);
        // Separador
        JSeparator jsResults = new JSeparator();
        jsResults.setOrientation(SwingConstants.HORIZONTAL);
        jsResults.setSize(getWidth() - 75, 2);
        jsResults.setLocation(25, txtClusters.getY() + txtClusters.getHeight() + 17);
        add(jsResults);
        // Titulo de resultados
        JLabel lblResults = new JLabel("Resultados");
        lblResults.setLocation(0, jsResults.getY() + jsResults.getHeight() + 5);
        lblResults.setSize(getWidth(), 24);
        lblResults.setHorizontalAlignment(JLabel.CENTER);
        lblResults.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblResults);
        // Epoca
        lblEpochResult = new JLabel("<html>Epoca: <b>0</b></html>");
        lblEpochResult.setLocation((int) ((lblSettings.getWidth() * 0.50) * 1.05), lblResults.getY() + lblResults.getHeight());
        lblEpochResult.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.90), 18);
        lblEpochResult.setHorizontalAlignment(JLabel.LEFT);
        lblEpochResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblEpochResult);
        // Error minimo
        lblErrorResult = new JLabel("<html>Error: <b>0.0</b></html>");
        lblErrorResult.setLocation((int) ((lblSettings.getWidth() * 0.50) * 0.05), lblResults.getY() + lblResults.getHeight());
        lblErrorResult.setSize((int) ((lblSettings.getWidth() * 0.50) * 0.90), 18);
        lblErrorResult.setHorizontalAlignment(JLabel.LEFT);
        lblErrorResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblErrorResult);
        // Grá fica del error
        errorSeries = new XYSeries("Error cuadratico");
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(errorSeries);
        // Grafica
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Error cuadratico por epoca en RBF",
                "Error",
                "Epoca",
                dataset,
                PlotOrientation.HORIZONTAL,
                true,
                true,
                false
        );
        // Obtenemos el ploteo para poder manipular la forma en la que se pintan
        XYPlot plot = chart.getXYPlot();
        // Creamos un nuevo render de línea para configurarlo nosotros.
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShape(0, ShapeUtilities.createDiamond(1.0f));
        try {
            StandardXYToolTipGenerator tooltipGenerator = new StandardXYToolTipGenerator()
            {
                @Override
                public String generateToolTip(XYDataset dataset, int series, int item)
                {
                    if ( errorSeries.getItems().size() != 0 ) {
                        return "Valor: " + dataset.getXValue(series, item);
                    }
                    return "";
                }
            };
            renderer.setBaseToolTipGenerator(tooltipGenerator);
        } catch ( Exception ignored) { }
        // Lo agregamos en nuestro gráfico
        plot.setRenderer(renderer);
        // Fondo blanco
        plot.setBackgroundPaint(Color.white);
        // Marca de rendijas (y)
        plot.setRangeGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.BLACK);
        // Marca de rendijas (x)
        plot.setDomainGridlinesVisible(true);
        plot.setDomainGridlinePaint(Color.BLACK);
        // Quitamos los bordes
        chart.getLegend().setBorder(BlockBorder.NONE);
        chart.setTitle(new TextTitle("Error cuadratico por epoca",
                        new Font("Dialog", Font.BOLD, 18)
                )
        );
        // Acomodo del gráfico dentro del JFrame
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setBackground(Color.white);
        chartPanel.setLocation(map.getX() + map.getWidth() + 35, map.getY());
        chartPanel.setSize(500, 400);
        add(chartPanel);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        setVisible(true);
    }

    private void changeUIState(boolean state) {
        btnRBF.setEnabled(state);
        btnForm1.setEnabled(state);
        btnForm2.setEnabled(state);
        btnForm3.setEnabled(state);
        txtClusters.setEnabled(state);
        txtEpochs.setEnabled(state);
        txtLearningRate.setEnabled(state);
        txtMinError.setEnabled(state);
        jmOptions.setEnabled(state);
        jmKmeans.setEnabled(state);
        jmRBF.setEnabled(state);
    }

    public void printCentroids(double[] centroids, boolean modelEnable) {
        double[] outputs = new double[centroids.length];
        Arrays.fill(outputs, 0);
        printCentroids(centroids, outputs, modelEnable);
    }

    public void printCentroids(double[] centroids, double[] outputs, boolean modelEnable) {
        ArrayList<Point> newPoints = new ArrayList<>();
        for ( Point point : points ) {
            if ( point.type != 3 ) {
                newPoints.add(point);
            }
        }
        points = newPoints;
        for ( int i = 0; i < centroids.length; i++ ) {
            Point point = new Point();
            point.x = centroids[i];
            point.y = outputs[i];
            point.xMap = ( point.x >= 0 ) ? (int) ((MAP_WIDTH * 0.5) + (point.x * ((MAP_WIDTH * 0.5) / MAP_SCALE))) : (int) ((MAP_WIDTH * 0.5) - (point.x * -((MAP_WIDTH * 0.5) / MAP_SCALE)));
            point.yMap = ( point.y >= 0 ) ? (int) ((MAP_HEIGHT * 0.5) - (point.y * ((MAP_HEIGHT * 0.5) / MAP_SCALE))) : (int) ((MAP_HEIGHT * 0.5) + (point.y * -((MAP_HEIGHT * 0.5) / MAP_SCALE)));
            point.color = Color.BLUE;
            point.type = 3;
            points.add(point);
        }
        if ( modelEnable ) {
            showSweep();
        } else {
            map.repaint();
        }
    }

    public void updateValuesForUI(double error, int epoch, boolean stop, boolean done) {
        errorSeries.add(error, epoch);
        if ( stop ) {
            if ( done ) {
                lblEpochResult.setText("<html>Convergio en la epoca: <b>" + epoch + "</b></html>");
            } else {
                lblEpochResult.setText("<html>No convergio. Epocas: <b>" + epoch + "</b></html>");
            }
        } else {
            lblEpochResult.setText("<html>Epoca: <b>" + epoch + "</b></html>");
        }
        lblErrorResult.setText("<html>Error: <b>" + error + "</b></html>");
    }

    public void setModel(RadialBasisNetworkThread.Model model, boolean print) {
        this.model = model;
        if ( print ) {
            System.out.println(model);
            changeUIState(true);
            clickable = true;
        }
        showSweep();
    }

    private void showSweep() {
        ArrayList<Point> newPoints = new ArrayList<>();
        for ( Point point : points ) {
            if ( point.type != 1 ) {
                newPoints.add(point);
            }
        }
        points = newPoints;
        for ( int i = 0; i < map.getWidth(); i++ ) {
            Point point = new Point();
            point.xMap = i;
            point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
            point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
            double y = 0;
            try {
                y = model.predict(new Object[] { point.x } );
            } catch (Exception e) {
                System.out.println("No se pudo realizar la prediccion del valor");
            }
            point.y = y;
            point.yMap = ( y >= 0 ) ? (int) ((MAP_HEIGHT * 0.5) - (y * ((MAP_HEIGHT * 0.5) / MAP_SCALE))) : (int) ((MAP_HEIGHT * 0.5) + (y * -((MAP_HEIGHT * 0.5) / MAP_SCALE)));
            point.color = Color.RED;
            point.type = 1;
            points.add(point);
        }
        map.repaint();
    }

    private class Map extends JPanel {

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            // Obtenemos el alto y ancho del componente
            int width = getWidth();
            int height = getHeight();
            // Linea vertical del lienzo
            g.drawLine(width / 2, 0, width / 2, height);
            // Linea horizontal
            g.drawLine(0, height / 2, width, height / 2);
            // ALgunas líneas más de apoyo
            g.setColor(new Color(170, 183, 184));
            for ( int i = 1; i < (width / MAP_SCALE / 8); i++ ) {
                int point = (int) (i * width / MAP_SCALE / 2);
                if ( point == width / 2 )  {
                    continue;
                }
                g.drawLine(point, 0, point, height);
                g.drawLine(0, point, width, point);
            }
            // Dibujamos los puntos hasta ahora obtenidos
            for ( int i = 0; i < points.size(); i++ ) {
                if ( points.get(i).type == 2 ) {
                    g.setColor(points.get(i).color);
                    if ( i != 0 && points.get(i - 1).type == 2 ) {
                        g.drawLine( points.get(i - 1).xMap, points.get(i - 1).yMap, points.get(i).xMap, points.get(i).yMap);
                    } else {
                        g.drawLine( points.get(i).xMap, points.get(i).yMap, points.get(i).xMap, points.get(i).yMap);
                    }
                } else if ( points.get(i).type == 1 ) {
                    g.setColor(points.get(i).color);
                    if ( i != 0 && points.get(i - 1).type == 1 ) {
                        g.drawLine( points.get(i - 1).xMap, points.get(i - 1).yMap, points.get(i).xMap, points.get(i).yMap);
                    } else {
                        g.drawLine( points.get(i).xMap, points.get(i).yMap, points.get(i).xMap, points.get(i).yMap);
                    }
                } else if ( points.get(i).type == 0 ) {
                    g.setColor(points.get(i).color);
                    g.fillOval(points.get(i).xMap - RADIUS_POINT, points.get(i).yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                } else if ( points.get(i).type == 3 ) {
                    g.setColor(points.get(i).color);
                    g.fillOval(points.get(i).xMap - RADIUS_POINT_CENTROID, points.get(i).yMap - RADIUS_POINT_CENTROID, RADIUS_POINT_CENTROID * 2, RADIUS_POINT_CENTROID * 2);
                }
            }
        }

    }

    private static class Point {

        public int xMap;
        public int yMap;
        public double x;
        public double y;
        public int type;
        public Color color;

        @Override
        public String toString() {
            return "Point{" +
                    "xMap=" + xMap +
                    ", yMap=" + yMap +
                    ", x=" + x +
                    ", y=" + y +
                    ", type=" + type +
                    ", color=" + color +
                    '}';
        }

    }

    private static class CustomKeyListener extends KeyAdapter {

        private final JTextField txtField;

        public CustomKeyListener(JTextField txtField)
        {
            this.txtField = txtField;
        }

        @Override
        public void keyTyped(KeyEvent e) {
            if ( (e.getKeyChar() < '0' || e.getKeyChar() > '9') && ( e.getKeyChar() != '.' && e.getKeyChar() != '-' ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '-' && ( txtField.getCaretPosition() != 0 || txtField.getText().contains("-") ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '.' && !txtField.getText().isEmpty() && ( txtField.getText().contains(".") ) ) {
                e.consume();
            }
            if ( txtField.getText().startsWith("-") && txtField.getCaretPosition() == 0 ) {
                e.consume();
            }
            super.keyTyped(e);
        }
    }

}
