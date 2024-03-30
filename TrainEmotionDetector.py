# import cv2
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.regularizers import l2

# train_data_gen = ImageDataGenerator(rescale=1./255)
# validation_data_gen = ImageDataGenerator(rescale=1./255)

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     shear_range=0.2,
#     rescale=1./255,
#     fill_mode='nearest')

# train_generator = train_data_gen.flow_from_directory(
#     'data/training',
#     target_size=(48, 48),
#     batch_size=32,
#     color_mode="grayscale",
#     class_mode='categorical',
#     shuffle=True
# )

# validation_generator = validation_data_gen.flow_from_directory(
#     'data/test',
#     target_size=(48, 48),
#     batch_size=32,
#     color_mode="grayscale",
#     class_mode='categorical',
#     shuffle=False
# )

# emotion_model = Sequential()

# #layer 1
# emotion_model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(48, 48, 1)))
# emotion_model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.5))

# #layer 2
# emotion_model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.5))

# #layer 3
# emotion_model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.5))

# emotion_model.add(Flatten())
# emotion_model.add(Dense(1024, activation='relu'))
# emotion_model.add(Dropout(0.5))
# emotion_model.add(Dense(2, activation='softmax'))

# cv2.ocl.setUseOpenCL(False)

# emotion_model.compile(
#     loss='categorical_crossentropy', 
#     optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
#     metrics=['accuracy'])

# checkpoint = ModelCheckpoint('model_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)

# ealy_stopping = EarlyStopping(monitor='accuracy',
#                               min_delta=0,
#                               patience=3,
#                               verbose=1,
#                               restore_best_weights=True)

# reduce_learningrate = ReduceLROnPlateau(monitor='accuracy',
#                                         factor=0.2,
#                                         patience=3,
#                                         verbose=1,
#                                         min_delta=0.0001)
# callbacks_list = [checkpoint]

# emotion_model_info = emotion_model.fit(
#         train_generator,
#         steps_per_epoch=480 // 32,
#         epochs=60,
#         validation_data=validation_generator,
#         validation_steps=120 // 32,
#         callbacks=callbacks_list)

# model_json = emotion_model.to_json()
# with open("emotion_model.json", "w") as json_file:
#     json_file.write(model_json)

# emotion_model.save_weights('emotion_model.h5')

# plt.figure(figsize=(20,10))
# plt.subplot(1, 2, 1)
# plt.suptitle('Optimizer : Adam', fontsize=10)
# plt.ylabel('loss', fontsize=16)
# plt.plot(emotion_model_info.history['loss'], label='Training Loss')
# plt.plot(emotion_model_info.history['val_loss'], label='Validation Loss')
# plt.legend(loc='upper right')

# plt.subplot(1, 2, 2)
# plt.ylabel('Accuracy', fontsize=16)
# plt.plot(emotion_model_info.history['accuracy'], label='Training Accuracy')
# plt.plot(emotion_model_info.history['val_accuracy'], label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.savefig('Pakai6.png')
# plt.show()


############## Confusion Matrix ###############
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, roc_curve, auc
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    rescale=1./255,
    fill_mode='nearest')

train_generator = train_data_gen.flow_from_directory(
    'data/training',
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

emotion_model = Sequential()

#layer 1
emotion_model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

#layer 2
emotion_model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

#layer 3
emotion_model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.2))
emotion_model.add(Dense(2, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(
    loss='categorical_crossentropy', 
    optimizer=Adam(learning_rate=0.0001, decay=1e-6), 
    metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)

ealy_stopping = EarlyStopping(monitor='accuracy',
                              min_delta=0,
                              patience=3,
                              verbose=1,
                              restore_best_weights=True)

reduce_learningrate = ReduceLROnPlateau(monitor='accuracy',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)
callbacks_list = [checkpoint]

emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=800 // 32,
        epochs=60,
        validation_data=validation_generator,
        validation_steps=200 // 32,
        callbacks=callbacks_list)

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('emotion_model.h5')

# Make predictions on the validation set
validation_steps = len(validation_generator)
validation_result = emotion_model.evaluate(validation_generator, steps=validation_steps)

# Get true labels and predicted labels
validation_generator.reset()
y_true = validation_generator.classes
y_pred = emotion_model.predict(validation_generator, steps=validation_steps, verbose=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('Confusion Matrix 4.png')
plt.show()

# Calculate and print classification report
class_report = classification_report(y_true, y_pred_labels, target_names=['Class 0', 'Class 1'])
print(class_report)

# Calculate and print AUC
roc_auc = roc_auc_score(y_true, y_pred_labels)
print(f'AUC: {roc_auc}')

# Calculate and print F1 score
f1 = f1_score(y_true, y_pred_labels)
print(f'F1 Score: {f1}')

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_labels)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.savefig('ROC 4.png')
plt.show()

# Plotting accuracy
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('loss', fontsize=16)
plt.plot(emotion_model_info.history['loss'], label='Training Loss')
plt.plot(emotion_model_info.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(emotion_model_info.history['accuracy'], label='Training Accuracy')
plt.plot(emotion_model_info.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.savefig('Accuracy 4.png')
plt.show()
