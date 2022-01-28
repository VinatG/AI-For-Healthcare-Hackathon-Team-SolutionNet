package com.example.ocular_disease_classification;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.ocular_disease_classification.ml.DiseaseQuantdi;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private ImageView imageView;
    private Button select,predict,search;
    private TextView textView;
    private Bitmap img;
    private TensorBuffer outputFeature0;
    private List<String> labels;
    private List<Float> output1;
    public float max = 0.0f;
    private String s2;
    public int index = 0;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.imageView);

        select = (Button) findViewById(R.id.button);
        predict = (Button) findViewById(R.id.button2);
        textView = (TextView) findViewById(R.id.textview);

        search = (Button) findViewById(R.id.search);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                textView.setText("");
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                                .add(getPostprocessNormalizeOp())
                                .build();
                img = Bitmap.createScaledBitmap(img,224,224,true);

                try {
                    DiseaseQuantdi model = DiseaseQuantdi.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                    tensorImage.load(img);
                    tensorImage = imageProcessor.process(tensorImage);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);
                    TensorBuffer inputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                    inputFeature1.loadBuffer(byteBuffer);
                    TensorBuffer inputFeature2 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                    inputFeature2.loadBuffer(byteBuffer);


                    // Runs model inference and gets result.

                    DiseaseQuantdi.Outputs outputs = model.process(inputFeature0, inputFeature1, inputFeature2);
                    outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    model.close();
                    for(int i=0;i<4;i++){
                        if(outputFeature0.getFloatArray()[i]>max){
                            max = outputFeature0.getFloatArray()[i];
                            index = i;
                        }
                    }
                    textView.setText(outputFeature0.getFloatArray()[0]+" "+outputFeature0.getFloatArray()[1]
                                        +" "+outputFeature0.getFloatArray()[2]+" "+outputFeature0.getFloatArray()[3]);
                    s2 = showResult(max,index);
                    max = 0;
                } catch (IOException e) {
                    e.getStackTrace();
                }
            }
        });

        search.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String searchURL = "https://en.wikipedia.org/wiki/" + s2;
                Uri earthquakeUri = Uri.parse(searchURL);

                Intent websiteIntent = new Intent(Intent.ACTION_VIEW,earthquakeUri);

                startActivity(websiteIntent);

            }
        });
    }

    private String showResult(float max,int index){
        try{
            labels = FileUtil.loadLabels(this,"label.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
        String name = labels.get(index);
        String out = labels.get(index) + "            "+String.valueOf(max);
        textView.setText(out);
        return name;
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100){
            String text = data.getData().toString();
            Log.d("fire12","text"+text);
            imageView.setImageURI(data.getData());
        }

        Uri uri = data.getData();
        try {
            img = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}