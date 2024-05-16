% Definir la ruta donde se guardan las imágenes
rutaCarpeta = 'ImagenesPreprocesadasUmbralizadas'; 

% Cargar imágenes utilizando imageDatastore
Imagenes = imageDatastore(rutaCarpeta, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Verificar el número de clases en los datos
etiquetas = unique(Imagenes.Labels);
numClases = numel(etiquetas);

% Mostrar el número de clases para verificación
fprintf('Número de clases detectadas: %d\n', numClases);

% Crear conjuntos de datos para entrenamiento y validación manualmente
DatosEntrenamientoFiles = [];
DatosValidacionFiles = [];
DatosEntrenamientoLabels = [];
DatosValidacionLabels = [];

for i = 1:numel(etiquetas)
    % Filtrar imágenes por etiqueta
    imgsPorEtiqueta = subset(Imagenes, Imagenes.Labels == etiquetas(i));
    filePaths = imgsPorEtiqueta.Files;
    
    % Tomar las primeras 2 imágenes como entrenamiento y las siguientes 2 como validación
    DatosEntrenamientoFiles = [DatosEntrenamientoFiles; filePaths(1:2)];
    DatosEntrenamientoLabels = [DatosEntrenamientoLabels; repmat(etiquetas(i), 2, 1)];
    DatosValidacionFiles = [DatosValidacionFiles; filePaths(3:4)];
    DatosValidacionLabels = [DatosValidacionLabels; repmat(etiquetas(i), 2, 1)];
end

% Crear imageDatastore para entrenamiento y validación
DatosEntrenamiento = imageDatastore(DatosEntrenamientoFiles, 'Labels', DatosEntrenamientoLabels);
DatosValidacion = imageDatastore(DatosValidacionFiles, 'Labels', DatosValidacionLabels);

% Guardar una copia de DatosValidacion para acceder a las etiquetas después
DatosValidacionOriginal = DatosValidacion;

% Definir el tamaño de entrada de la red para imágenes en escala de grises
inputSize = [400 400 1];

% Crear augmentedImageDatastore para redimensionar las imágenes
DatosEntrenamiento = augmentedImageDatastore(inputSize, DatosEntrenamiento);
DatosValidacion = augmentedImageDatastore(inputSize, DatosValidacion);

% Definir la arquitectura de la red neuronal convolucional
capas = [
    imageInputLayer(inputSize) % Tamaño ajustado al tamaño de las imágenes

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClases) % Ajustar el número de clases al número detectado
    softmaxLayer
    classificationLayer];

% Especificar las opciones de entrenamiento
opciones = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', DatosValidacion, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Entrenar la red neuronal
red = trainNetwork(DatosEntrenamiento, capas, opciones);

% Clasificar las imágenes de validación y calcular la precisión
predicciones = classify(red, DatosValidacion);
etiquetasValidacion = DatosValidacionOriginal.Labels;

% Crear la matriz de confusión visual
figura = figure;
matrizConfusion = confusionchart(etiquetasValidacion, predicciones);

% Personalizar la matriz de confusión
matrizConfusion.Title = 'Matriz de Confusión de la Clasificación';
matrizConfusion.ColumnSummary = 'column-normalized';
matrizConfusion.RowSummary = 'row-normalized';

% Calcular y mostrar la precisión
precision = sum(predicciones == etiquetasValidacion) / numel(etiquetasValidacion);
fprintf('Precisión: %.2f%%\n', precision * 100);

% Guardar la red entrenada
save('redLenguajeSenasEntrenada.mat', 'red');
