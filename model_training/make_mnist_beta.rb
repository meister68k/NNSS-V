#!/usr/bin/env ruby
# -*- coding: utf-8 -*-
#
# MNISTのべたイメージを生成

require 'numo/narray'

require 'pp'

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

# MNISTデータセットの準備
mnist = MNIST.new
mnist.load_data('../MNIST/train-images-idx3-ubyte', '../MNIST/train-labels-idx1-ubyte')

ARGV.each {|param|
    eval(param).each {|idx|
        label = mnist.get_label(idx)
        dat = mnist.get_image(idx).to_a.flatten.pack('C*')
        open("IMG_#{idx}_#{label}.BIN", 'w') {|ofs| ofs.write(dat)}
    }
}
