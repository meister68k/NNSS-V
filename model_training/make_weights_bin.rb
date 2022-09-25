#!/usr/bin/env ruby
# -*- coding: utf-8 -*-
#
# 重みのべたバイナリデータを生成

require 'json'
require 'pp'

weights = JSON.load(open('../prototype3/MNIST_uint8.json').read)

bin_weights = 
    weights['conv2d']['bias'].flatten.pack('C*') +
    weights['conv2d']['kernel'].flatten.pack('C*') +
    weights['dense']['kernel'].flatten.pack('C*')

p bin_weights.size

open('MNMODEL.BIN', 'wb') {|ofs| ofs.write(bin_weights)}
