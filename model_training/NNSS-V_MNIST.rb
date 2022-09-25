#!/usr/bin/env ruby
# -*- coding: utf-8 -*-
#
# Neural Network Simulation System ver 5 -- 8bit量子化
#
# 2021-01-18 NAKAUE,T
#
# acc 実数 94.7%
# 8bit(3/5bit)量子化後 94.7%
# bias省略 93.5%
#
#Model: "sequential"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dropout (Dropout)            (None, 28, 28, 1)         0         
#_________________________________________________________________
#lambda (Lambda)              (None, 28, 28, 1)         0         
#_________________________________________________________________
#conv2d (Conv2D)              (None, 13, 13, 8)         80        
#_________________________________________________________________
#flatten (Flatten)            (None, 1352)              0         
#_________________________________________________________________
#dense (Dense)                (None, 10)                13520     
#=================================================================
#Total params: 13,600
#Trainable params: 13,600
#Non-trainable params: 0
#_________________________________________________________________

require 'chunky_png'
require 'numo/narray'

require 'json'
require 'matrix'
require 'pp'


####################
# ASCIIアート出力
# intなら0～255，floatなら0～1.0とする
def aa_puts(ary, scale = nil)
    # ASCIIアート用の濃淡マップ
    aamap = '.:!*%$@&#SB'

    ary = ary.to_a if !ary.kind_of?(Array)

    if !scale
        scale = 1.0 * aamap.size
        scale = scale / 255.0 if ary[0][0].kind_of?(Integer)
    end

    ary.each {|line|
        puts line.map{|x| aamap[(x * scale).to_i.clamp(0, aamap.size - 1)]}.join
    }
end

####################
# ASCIIアートグラフ
# intなら0～255，floatなら0～1.0とする
def aa_graph(ary, width = 20, scale = nil)
    ary = ary.to_a if !ary.kind_of?(Array)

    if !scale
        scale = 1.0 * width
        scale = scale / 255.0 if ary[0].kind_of?(Integer)
    end

    ary.each_with_index {|x, i|
        puts format('% 2d : %s %.2f', i, '*' * (x * scale), x)
    }
end

####################
# MNISTデータの読込み
#
class MNIST
    attr_accessor :image                # 画像(バイナリで保持)
    attr_accessor :label                # ラベル(バイナリで保持)
    attr_accessor :width                # 画像の幅
    attr_accessor :height               # 画像の高さ
    attr_accessor :len                  # 画像のバイト数
    attr_accessor :size                 # 画像の枚数


    def load_data(image_binary, label_binary)
        open(label_binary, 'rb') {|ifs|
            ifs.read(8)                 # ヘッダ読み飛ばし
            @label = ifs.read
        }

        open(image_binary, 'rb') {|ifs|
            ifs.read(4)                 # ヘッダ読み飛ばし
            tmp , @height , @width = ifs.read(12).unpack('N*')
            @len = @height * @width
            @image = ifs.read

            @size = @image.size / @len
        }

    end


    def get_label(idx)
        @label[idx].unpack('C').first
    end


    def get_image(idx)
        Numo::UInt8.cast(@image[(idx * @len) ... ((idx+1) * @len)].unpack('C*')).reshape(@height, @width)
    end


    def get_image_float(idx)
        Numo::SFloat.cast(get_image(idx)) / 255.0
    end

end

####################
# Softmax
#
def softmax(x)
    tmp = Numo::NMath.exp(x)
    return tmp / tmp.sum
end

####################
# ReLU
#
def relu(x)
    [x, x * 0].max                      # xと同じ型にするため小細工
end

####################
# MNIST専用ネットワーク(整数演算)
#
class NetworkMNIST_UInt8
    attr_accessor :weights              # 重み

    def initialize
        @weights = {}
        @input_shape = [28, 28, 1]
        @cnn_shape = [13, 13, 1, 8]
    end


    # JSONからの重みの読込み(実数)
    def load_weights_float(json_str)
        dat = JSON.load(json_str)

        [
            ['conv2d/kernel:0', :conv2d, :kernel, 32 / 255.0 * 256],
            ['conv2d/bias:0',   :conv2d, :bias, 32],
            ['dense/kernel:0',  :dense,  :kernel, 32],
            ['dense/bias:0',    :dense,  :bias, 32],
        ].each{|name, layer, key, scale|
            next if !dat.key?(name)
            @weights[layer] = {} if !@weights[layer]
            @weights[layer][key] = Numo::Int8.cast((Numo::SFloat.cast(dat[name])) * scale)
        }
    end


    # JSONからの重みの読込み
    def load_weights(json_str)
        @weights = JSON.load(json_str).map {|layer, val|
            [
                layer.to_sym,
                val.map {|key, ary| [key.to_sym, Numo::Int8.cast(ary)]}.to_h
            ]
        }.to_h
    end


    # JSON形式の重みの保存
    def save_weights()
        JSON.dump(@weights.map {|layer, val|
            [
                layer,
                val.map {|key, ary| [key, ary.to_a]}.to_h
            ]
        }.to_h)
    end


    # 入力層
    def input_layer(input)
        return Numo::UInt8.cast(input).reshape(*(@input_shape))
    end


    # CNN層
    def conv2d_layer(input)
        output = Numo::Int16.zeros(@cnn_shape)

        @cnn_shape[3].times {|k|
            kernel = @weights[:conv2d][:kernel][true, true, 0, k]
            bias = @weights[:conv2d][:bias][k]

            @cnn_shape[1].times {|j|
                y = j * 2
                @cnn_shape[0].times {|i|
                    x = i * 2
                    inp = Numo::Int16.cast(input[y..(y + 2), x..(x + 2), 0])
                    logits = (inp * kernel).sum / 256 + bias
                    output[j, i, 0, k] = Numo::UInt8.cast(relu(logits).clamp(0, 255))
                }
            }
        }

        return output
    end


    # フラット化層
    def flatten_layer(input)
        return input.reshape(1, input.size)
    end


    # 全結合層
    def dense_layer(input, use_bias: true)
        kernel = @weights[:dense][:kernel]
        if use_bias
            bias = @weights[:dense][:bias]
        else
            bias = Numo::Int8.zeros(kernel.shape[-1])
        end
        logits = (Numo::Int16.cast(input).dot kernel) / 32 + bias

        return logits.to_a.flatten.map{|x| x.clamp(0, 255)}
    end


    # 予測値の計算
    def predict(input)
        output_1 = input_layer(input)
        output_2 = conv2d_layer(output_1)
        output_3 = flatten_layer(output_2)
        output_4 = dense_layer(output_3, use_bias: false)

        return output_4
    end

end
####################

# ネットワークモデルの初期化
model = NetworkMNIST_UInt8.new
model.load_weights_float(open('MNIST_minimum2.json').read)
open('MNIST_uint8.json','w') {|ofs| ofs.write(model.save_weights)}
model.load_weights(open('MNIST_uint8.json').read)

# MNISTデータセットの準備
valid = MNIST.new
valid.load_data('../MNIST/train-images-idx3-ubyte', '../MNIST/train-labels-idx1-ubyte')

=begin
(2..10).each {|i|
    input = valid.get_image(i)
    collect = valid.get_label(i)

    predict = model.predict(input)
    answer = predict.each_with_index.max[1]

    puts "#{i} : Input = #{collect} Output = #{answer}"
    pp predict

    #aa_puts input
    #aa_graph(predict, 40)
}
exit
=end

# 評価
accuracy = 0
valid.size.times {|i|
    input = valid.get_image(i)
    collect = valid.get_label(i)

    predict = model.predict(input)
    answer = predict.each_with_index.max[1]

    accuracy += ((collect == answer) ? 1 : 0)

    puts "#{i} : Input = #{collect} Output = #{answer} Accuracy = #{format('%.1f', accuracy.to_f / (i + 1) * 100.0)}%"

    #aa_puts input
    #aa_graph(predict, 40)
}


