package com.example.classificationofdeep_fake;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.classificationofdeep_fake.ml.Model;
import com.example.classificationofdeep_fake.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;

public class MainActivity extends AppCompatActivity {
    Button selectBtn;
    TextView predictionResult;
    Bitmap bitmap;
    ImageView imageView;
    int imageSize = 224;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        selectBtn= findViewById(R.id.selectBtn);
        predictionResult= findViewById(R.id.Result);
        imageView= findViewById(R.id.imageView);
        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });


    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==10){
            if(data!=null){
                Uri uri= data.getData();
                try {
                    bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    imageView.setImageBitmap(bitmap);
                    bitmap = Bitmap.createScaledBitmap(bitmap,imageSize,imageSize, false);
                    classify(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
    public void classify(Bitmap image){
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4* imageSize* imageSize* 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intValue = new int [imageSize*imageSize];
            bitmap.getPixels(intValue,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
            int pixel=0;
            for (int i=0; i<imageSize;i++){
                for (int j=0;j<imageSize;j++){
                    int val= intValue[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences=outputFeature0.getFloatArray();
            final DecimalFormat df = new DecimalFormat("0.00");

            if (confidences[0] > confidences[1]) {
                predictionResult.setText("REAL IMAGE \n real-"+confidences[0]+"\nfake-"+confidences[1]);
            } else {
                predictionResult.setText("FAKE IMAGE \n real-"+confidences[0]+"\nfake-"+confidences[1]);
            }

            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }
}