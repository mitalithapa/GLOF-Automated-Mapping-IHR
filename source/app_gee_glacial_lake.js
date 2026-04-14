/**
 * Automated GLOF Mapping & Monitoring Prototype
 * * Purpose: Automated detection of glacial lake boundaries in the IHR.
 * Methodology: Random Forest (RF) trained on Sentinel-2 MSI + Topographic Features.
 * * Key Mitigations:
 * 1. Shadow Masking: Uses Slope & Hillshade derived from Hybrid ALOS DEM.
 * 2. Snow Masking: NDSI filtering to separate frozen lakes from open water.
 * 3. Dynamic AOI: Real-time calculation of lake surface area.
 * * Author: Defense Organization Research Scientist
 */

// =================================================================
// 1. SYSTEM INITIALIZATION & UI SETUP
// =================================================================

// Modern Enterprise Web App Color Palette
var theme = {
  sidebarBg: '#f8fafc', // Slate 50
  cardBg: '#ffffff',
  headerBg: '#0f172a',  // Slate 900
  headerText: '#f8fafc',
  textMain: '#1e293b',  // Slate 800
  textMuted: '#64748b', // Slate 500
  border: '#e2e8f0',    // Slate 200
  accent: '#3b82f6',    // Blue 500
  accentHover: '#2563eb',// Blue 600
  danger: '#dc2626',    // Red 600
  dangerBg: '#fef2f2',
  dangerBorder: '#fecaca',
  success: '#16a34a',   // Green 600
  successBg: '#f0fdf4',
  successBorder: '#bbf7d0',
  infoBg: '#eff6ff',
  infoBorder: '#bfdbfe',
  terminalBg: '#020617', // Slate 950
  terminalText: '#22c55e'// Green 500
};

var cardStyle = {
  backgroundColor: theme.cardBg, 
  margin: '10px 16px', 
  padding: '16px', 
  border: '1px solid ' + theme.border
};

// Root Sidebar Panel
var mainPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {width: '420px', height: '100%', backgroundColor: theme.sidebarBg, border: 'none'}
});

// High-Contrast SaaS Header
var headerPanel = ui.Panel({
  widgets: [
    ui.Panel([
      ui.Label('🛰️', {fontSize: '24px', backgroundColor: theme.headerBg, margin: '0 8px 0 0'}),
      ui.Label('GLOF WATCH', {fontWeight: 'bold', fontSize: '24px', color: theme.headerText, backgroundColor: theme.headerBg, margin: '0'})
    ], ui.Panel.Layout.Flow('horizontal'), {backgroundColor: theme.headerBg, margin: '20px 16px 4px 16px'}),
    
    ui.Label('v2.5 • Tactical Himalayan Glacial Monitor', {
      fontSize: '11px', color: '#94a3b8', fontFamily: 'monospace',
      backgroundColor: theme.headerBg, margin: '0 16px 20px 16px'
    })
  ],
  style: {backgroundColor: theme.headerBg, stretch: 'horizontal', margin: '0'}
});

// Simulated bottom border using a colored label to avoid GEE style property errors
var headerDivider = ui.Label('', {
  backgroundColor: theme.accent, 
  stretch: 'horizontal', 
  height: '3px', 
  margin: '0 0 8px 0',
  padding: '0'
});

mainPanel.add(headerPanel);
mainPanel.add(headerDivider);

var map = ui.Map();
map.setOptions('SATELLITE');
map.style().set('cursor', 'crosshair');
map.setControlVisibility({layerList: false, mapTypeControl: false}); // Clean up default map UI

map.drawingTools().setShown(true);
map.drawingTools().setLinked(false);

ui.root.clear();
ui.root.add(mainPanel);
ui.root.add(map);

// =================================================================
// 2. CONFIGURATION & SITES
// =================================================================

var sites = {
  'Baralacha La (Lahaul)': [77.4201, 32.7585],
  'Samundar Tapu': [77.5400, 32.4999],
  'Khangchengyao (Sikkim)': [88.6551, 27.9850],
  'Karzok (Ladakh)': [78.2640, 32.9681],
  'Zullu Lake': [77.4124, 32.6105]
};

var selectSite = ui.Select({
  items: Object.keys(sites),
  placeholder: 'Select Target Monitoring Site...',
  style: {stretch: 'horizontal', margin: '8px 0 0 0'},
  onChange: function(key) {
    map.setCenter(sites[key][0], sites[key][1], 13);
  }
});

var step1 = ui.Panel({
  widgets: [
    ui.Label('01 | OBSERVATION BASIN', {fontWeight: 'bold', fontSize: '11px', color: theme.accent, margin: '0 0 4px 0'}),
    ui.Label('Select a known high-risk glacier basin or draw a custom polygon on the map.', {fontSize: '12px', color: theme.textMuted, margin: '0'}),
    selectSite
  ],
  style: cardStyle
});

var startYear = ui.Textbox({value: '2023-07-01', style: {width: '120px', margin: '0 8px 0 0'}});
var endYear = ui.Textbox({value: '2023-10-31', style: {width: '120px', margin: '0 0 0 8px'}});

var step2 = ui.Panel({
  widgets: [
    ui.Label('02 | TEMPORAL PARAMETERS', {fontWeight: 'bold', fontSize: '11px', color: theme.accent, margin: '0 0 4px 0'}),
    ui.Panel([
      ui.Label('Start:', {margin: '4px 4px 0 0', fontSize: '12px', color: theme.textMain}), startYear, 
      ui.Label('End:', {margin: '4px 4px 0 0', fontSize: '12px', color: theme.textMain}), endYear
    ], ui.Panel.Layout.Flow('horizontal'))
  ],
  style: cardStyle
});

var mndwiThreshold = ui.Textbox({value: '0.15', style: {width: '60px', margin: '0 0 0 8px'}});
var step3 = ui.Panel({
  widgets: [
    ui.Label('03 | ADVANCED TUNING', {fontWeight: 'bold', fontSize: '11px', color: theme.accent, margin: '0 0 4px 0'}),
    ui.Panel([
      ui.Label('Heuristic Fallback MNDWI Threshold:', {fontSize: '11px', color: theme.textMuted, margin: '4px 0 0 0'}), 
      mndwiThreshold
    ], ui.Panel.Layout.Flow('horizontal'))
  ],
  style: cardStyle
});

mainPanel.add(step1).add(step2).add(step3);

// =================================================================
// 3. SCIENTIFIC CORE: FEATURE ENGINEERING
// =================================================================

function maskClouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)
    .copyProperties(image, ['system:time_start', 'MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE']);
}

function extractFeatures(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI'); 
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI'); 
  var ndsi = image.normalizedDifference(['B3', 'B11']).rename('NDSI'); 
  
  var globalDem = ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').mosaic().select('DSM').toFloat();
  
  var customDemCollection = ee.ImageCollection([
    ee.Image("users/mitalithapa07/barlachla2"), ee.Image("users/mitalithapa07/beas"),
    ee.Image("users/mitalithapa07/bhaga"), ee.Image("users/mitalithapa07/bhaga13"),
    ee.Image("users/mitalithapa07/chandra"), ee.Image("users/mitalithapa07/fanchan"),
    ee.Image("users/mitalithapa07/gepang"), ee.Image("users/mitalithapa07/hangu"),
    ee.Image("users/mitalithapa07/kaktital"), ee.Image("users/mitalithapa07/karzok2"),
    ee.Image("users/mitalithapa07/lamdalravi"), ee.Image("users/mitalithapa07/langpo2"),
    ee.Image("users/mitalithapa07/neelkanth"), ee.Image("users/mitalithapa07/paldanlamotal"),
    ee.Image("users/mitalithapa07/ravibasin"), ee.Image("users/mitalithapa07/samundraTapu"),
    ee.Image("users/mitalithapa07/tsoparidhi"), ee.Image("users/mitalithapa07/vasuki"),
    ee.Image("users/mitalithapa07/zullu"), ee.Image("users/mitalithapa07/zullu2")
  ]);
  
  var uniformCustomDem = customDemCollection.map(function(img) { return img.select([0]).rename('DSM').toFloat(); });
  var dem = ee.ImageCollection([globalDem, uniformCustomDem.mosaic()]).mosaic().rename('DSM');
  var slope = ee.Terrain.slope(dem).rename('Slope');
  
  var azimuth = ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE'));
  var zenith = ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE'));
  var altitude = ee.Number(90).subtract(zenith);
  var hillshade = ee.Terrain.hillshade(dem, azimuth, altitude).rename('Hillshade');
  
  return image.addBands([ndwi, mndwi, ndsi, dem, slope, hillshade]);
}

// =================================================================
// 4. CLASSIFICATION & UI LOGIC
// =================================================================

var runContainer = ui.Panel({style: {backgroundColor: theme.sidebarBg, padding: '4px 16px'}});

var runButton = ui.Button({
  label: '▶ INITIATE SATELLITE SCAN',
  style: {stretch: 'horizontal', color: theme.headerBg, fontWeight: 'bold'}
});
runContainer.add(runButton);
mainPanel.add(runContainer);

// Telemetry Terminal Panel
var terminalPanel = ui.Panel({
  layout: ui.Panel.Layout.flow('vertical'),
  style: {backgroundColor: theme.terminalBg, padding: '8px', margin: '0 16px 12px 16px', height: '100px', border: '1px solid #334155'}
});
mainPanel.add(terminalPanel);

function logTerminal(msg) {
  var time = new Date().toLocaleTimeString();
  terminalPanel.add(ui.Label('['+time+'] ' + msg, {
    color: theme.terminalText, fontSize: '10px', fontFamily: 'monospace', 
    backgroundColor: theme.terminalBg, margin: '2px 0'
  }));
}

var reportContainer = ui.Panel({style: {padding: '0', margin: '0', backgroundColor: theme.sidebarBg}});
mainPanel.add(reportContainer);

runButton.onClick(function() {
  terminalPanel.clear();
  reportContainer.clear();
  runButton.setDisabled(true);
  
  logTerminal('Establishing connection to Earth Engine...');
  
  var roi;
  var dtLayers = map.drawingTools().layers();
  var userDrewGeometry = false;
  
  for (var i = 0; i < dtLayers.length(); i++) {
    if (dtLayers.get(i).geometries().length() > 0) {
      userDrewGeometry = true;
      break;
    }
  }

  if (userDrewGeometry) {
    roi = map.drawingTools().toFeatureCollection().geometry();
    logTerminal('Custom bounding geometry acquired.');
  } else {
    var bounds = map.getBounds();
    roi = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]]);
    logTerminal('Viewport mapped to processing coordinates.');
  }
  
  logTerminal('Querying Sentinel-2 Harmonized MSI collection...');
  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startYear.getValue(), endYear.getValue())
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    .map(maskClouds)
    .map(extractFeatures);
    
  var composite = s2Col.median().clip(roi);
  
  logTerminal('Applying topographically corrected feature engineering...');
  var trainingAssets = [
    "users/mitalithapa07/zullu_aoi", "users/mitalithapa07/vasukiLake_aoi",
    "users/mitalithapa07/samundratapuaoi", "users/mitalithapa07/ravibasin_aoi",
    "users/mitalithapa07/bhaga_aoi", "users/mitalithapa07/beas_aoi","users/mitalithapa07/Khanchengyao_aoi",
    "users/mitalithapa07/barlachlaaoi","users/mitalithapa07/bhaga12_aoi","users/mitalithapa07/bhaga13_aoi",
    "users/mitalithapa07/chandra_aoi","users/mitalithapa07/fanchan2_aoi", "users/mitalithapa07/fanchan_aoi",
    "users/mitalithapa07/gepangghat_aoi","users/mitalithapa07/hangu_aoi","users/mitalithapa07/kaktital_aoi",
    "users/mitalithapa07/karzok_aoi","users/mitalithapa07/lamdalravi_aoi","users/mitalithapa07/langpopeak_aoi",
    "users/mitalithapa07/neelkanth_aoi","users/mitalithapa07/paldanlamotal_aoi","users/mitalithapa07/tsoparidhi2_aoi",
    "users/mitalithapa07/tsoparidhi_aoi","users/mitalithapa07/vasuki2_aoi","users/mitalithapa07/vasuki3_aoi",
    "users/mitalithapa07/vasuki4_aoi"
  ];
  
  var rawTrainingData = ee.FeatureCollection(
    trainingAssets.map(function(asset) { return ee.FeatureCollection(asset); })
  ).flatten().filterBounds(roi);
  
  var trainingData = rawTrainingData.map(function(feature) {
    var c1 = feature.get('class');
    var c2 = feature.get('Class');
    var c3 = feature.get('CLASS');
    var finalClass = ee.Algorithms.If(ee.Algorithms.IsEqual(c1, null),
                       ee.Algorithms.If(ee.Algorithms.IsEqual(c2, null), c3, c2), c1);
    return feature.set('class', finalClass);
  }).filter(ee.Filter.notNull(['class']));
  
  var sampleCount = trainingData.size();

  sampleCount.evaluate(function(count) {
    var classified;
    var modelAccuracy = ee.Number(-1);

    if (count > 0) {
      logTerminal('Bootstrapping Random Forest (250 trees)...');
      var classifier = ee.Classifier.smileRandomForest({
        numberOfTrees: 250, 
        variablesPerSplit: 7, 
        minLeafPopulation: 7
      }).train({
        features: composite.sampleRegions({
          collection: trainingData, properties: ['class'], scale: 10, tileScale: 4
        }),
        classProperty: 'class',
        inputProperties: ['B2', 'B3', 'B4', 'B8', 'B11', 'NDWI', 'MNDWI', 'NDSI', 'Slope', 'Hillshade']
      });
      classified = composite.classify(classifier).eq(1);
      modelAccuracy = classifier.confusionMatrix().accuracy();
    } else {
      var thresh = parseFloat(mndwiThreshold.getValue());
      logTerminal('WARN: Zero training polygons. Engaging MNDWI heuristic (>' + thresh + ').');
      classified = composite.select('MNDWI').gt(thresh)
                   .and(composite.select('Slope').lt(10))
                   .and(composite.select('NDSI').lt(0.3));
    }
    
    var terrainMask = composite.select('Slope').lt(20);
    var refinedLakes = classified.updateMask(terrainMask).selfMask();
    
    // VISUALIZATION & TOGGLES
    map.clear();
    
    if (userDrewGeometry) {
      var roiOutline = ee.Image().byte().paint({featureCollection: ee.FeatureCollection([ee.Feature(roi)]), color: 1, width: 2});
      map.addLayer(roiOutline, {palette: ['#FF0000']}, 'Custom AOI Boundary');
    }

    var baseLayer = map.addLayer(composite, {bands: ['B11', 'B8', 'B4'], min: 0, max: 0.3}, 'Sentinel-2 SWIR/NIR/R');
    var lakeLayer = map.addLayer(refinedLakes, {palette: ['#00ffff']}, 'Detected Glacial Lake Boundary');
    
    // Add Interactive Map Toggles
    var layerControls = ui.Panel({
      style: {position: 'bottom-left', padding: '8px', backgroundColor: 'rgba(255,255,255,0.9)', border: '1px solid ' + theme.border}
    });
    layerControls.add(ui.Label('LAYERS', {fontWeight: 'bold', fontSize: '10px', color: theme.textMuted, margin: '0 0 4px 0'}));
    
    var toggleS2 = ui.Checkbox('Sentinel-2 Base', true);
    toggleS2.onChange(function(checked) { baseLayer.setShown(checked); });
    var toggleLake = ui.Checkbox('Lake Mask (Cyan)', true);
    toggleLake.onChange(function(checked) { lakeLayer.setShown(checked); });
    
    layerControls.add(toggleS2).add(toggleLake);
    map.add(layerControls);

    logTerminal('Calculating spatial geometries...');
    var areaImage = refinedLakes.multiply(ee.Image.pixelArea()).rename('area');
    var rawArea = areaImage.reduceRegion({
      reducer: ee.Reducer.sum(), geometry: roi, scale: 10, maxPixels: 1e10, bestEffort: true
    }).get('area');
    
    var payloadDict = ee.Dictionary({
      'area': ee.Algorithms.If(ee.Algorithms.IsEqual(rawArea, null), 0, rawArea),
      'accuracy': modelAccuracy
    });
    
    payloadDict.evaluate(function(result) {
      var areaKm2 = (result && result.area != null) ? (result.area / 1e6) : 0;
      var accuracy = (result && result.accuracy != null) ? result.accuracy : -1;
      
      logTerminal('Scan complete. Generating payload reports.');
      runButton.setDisabled(false);
      
      // DASHBOARD RESULTS CARD
      var resultsCard = ui.Panel({style: cardStyle});
      
      var headerRow = ui.Panel([
        ui.Label('SCAN COMPLETE', {fontWeight: 'bold', fontSize: '11px', color: theme.success, margin: '0'})
      ], ui.Panel.Layout.Flow('horizontal'));
      resultsCard.add(headerRow);
      
      var metricPanel = ui.Panel({
        widgets: [
          ui.Label('TOTAL SURFACE AREA', {fontSize: '10px', fontWeight: 'bold', color: theme.textMuted, margin: '16px 0 0 0'}),
          ui.Label(areaKm2.toFixed(3) + ' km²', {fontSize: '36px', fontWeight: 'bold', color: theme.textMain, margin: '4px 0 16px 0'})
        ],
        style: {backgroundColor: theme.cardBg}
      });
      resultsCard.add(metricPanel);
      
      // Dynamic Chart: Spectral Signature Profile
      if (areaKm2 > 0) {
        var spectralBands = composite.select(['B2', 'B3', 'B4', 'B8', 'B11']).updateMask(refinedLakes);
        
        // Define approximate central wavelengths for X-axis
        var wavelengths = [490, 560, 665, 842, 1610];
        
        // Use an arbitrary sample point within the lake to plot the signature
        var signatureChart = ui.Chart.image.regions({
          image: spectralBands,
          regions: roi,
          reducer: ee.Reducer.mean(),
          scale: 30,
          xLabels: wavelengths
        }).setChartType('LineChart').setOptions({
          title: 'Mean Spectral Reflectance of Detected Water',
          titlePosition: 'none',
          hAxis: {title: 'Wavelength (nm)', titleTextStyle: {italic: false, fontSize: 10}},
          vAxis: {title: 'Reflectance', titleTextStyle: {italic: false, fontSize: 10}},
          colors: ['#0891b2'],
          lineWidth: 2,
          pointSize: 4,
          legend: {position: 'none'},
          height: 150,
          chartArea: {width: '75%', height: '65%'}
        });
        
        var chartPanel = ui.Panel({style: {margin: '8px 0', border: '1px solid ' + theme.border}});
        chartPanel.add(ui.Label('SPECTRAL SIGNATURE PROFILE', {fontWeight: 'bold', fontSize: '10px', color: theme.accent, margin: '8px 8px 0 8px'}));
        chartPanel.add(signatureChart);
        resultsCard.add(chartPanel);
      }
      
      var transparencyCard = ui.Panel({
        style: {padding: '12px', margin: '8px 0', backgroundColor: theme.infoBg, border: '1px solid ' + theme.infoBorder}
      });
      transparencyCard.add(ui.Label('MODEL TRANSPARENCY', {fontWeight: 'bold', fontSize: '10px', color: theme.accent, margin: '0 0 4px 0'}));
      
      if (accuracy >= 0) {
        var accPercent = (accuracy * 100).toFixed(2);
        transparencyCard.add(ui.Label('Trained dynamically using local spectral profiles. Model Accuracy: ' + accPercent + '%', {fontSize: '12px', color: '#1e40af', margin: '0'}));
      } else {
        transparencyCard.add(ui.Label('No local training polygons found. Using heuristic threshold fallback.', {fontSize: '12px', color: theme.danger, margin: '0'}));
        transparencyCard.style().set('backgroundColor', theme.dangerBg);
        transparencyCard.style().set('border', '1px solid ' + theme.dangerBorder);
      }
      resultsCard.add(transparencyCard);
      
      var alertPanel = ui.Panel({style: {padding: '12px', margin: '8px 0', border: '1px solid ' + theme.cardBg}});
      if (areaKm2 > 0.1) {
         alertPanel.style().set('backgroundColor', theme.dangerBg);
         alertPanel.style().set('border', '1px solid ' + theme.dangerBorder);
         alertPanel.add(ui.Label('HIGH VOLUME ALERT', {color: theme.danger, fontWeight: 'bold', fontSize: '11px', margin: '0 0 4px 0'}));
         alertPanel.add(ui.Label('Significant lake volume detected. Compare against historical baseline to assess expansion rate.', {fontSize: '12px', color: '#991b1b', margin: '0'}));
      } else {
         alertPanel.style().set('backgroundColor', theme.successBg);
         alertPanel.style().set('border', '1px solid ' + theme.successBorder);
         alertPanel.add(ui.Label('STATUS: NOMINAL', {color: theme.success, fontWeight: 'bold', fontSize: '11px', margin: '0 0 4px 0'}));
         alertPanel.add(ui.Label('Lake volume is within normal monitoring thresholds.', {fontSize: '12px', color: '#166534', margin: '0'}));
      }
      resultsCard.add(alertPanel);
      
      reportContainer.add(resultsCard);
    });
  });
});
