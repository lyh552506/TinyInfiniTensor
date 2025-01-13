#include "../../include/operators/unary.h"
#include <limits>

namespace infini
{
    UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
        : OperatorObj(type, {input}, {output})
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        return {{A->getDims()}};
    }

    std::string UnaryObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                     std::optional<float> min, std::optional<float> max)
        : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
          maxValue(max)
    {
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 clip 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13
        // =================================== 作业 ===================================
		const auto A=inputs[0];
		auto input_dim=A->getDims();
		auto output_dim = input_dim;
		auto clip_min=std::numeric_limits<int>::lowest();
		auto clip_max=std::numeric_limits<int>::max();
		if(this->getMax().has_value())
		  clip_max=this->getMax().value();
		if(this->getMin().has_value())
		  clip_min=this->getMin().value();
		if(clip_min>clip_max){
			for(auto &ind:output_dim)
				ind=clip_max;
			return std::optional<vector<Shape>>({output_dim});
		}
		for(auto &ind:output_dim){
			if(ind<clip_min)
			  {ind=clip_min;continue;}
			if(ind>clip_max)
			  {ind=clip_max;continue;}
		}
        return std::optional<vector<Shape>>({output_dim});
    }

    std::string ClipObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
        : OperatorObj(OpType::Cast, {input}, {output}), castType(type)
    {
        IT_ASSERT(checkValid(graph));
    }

    vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后, 输出 tensor 的数目和数据类型
        // REF_FILE: src/core/operator.cc
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
        return {};
    }

    optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 cast 操作后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Cast.html#cast-21
        // =================================== 作业 ===================================
		const auto A=inputs[0];
		auto input_dims=A->getDims();
		auto output_dims=input_dims;
		switch(castType){
			case infini::CastType::Float2Float16:

			break;
    case infini::CastType::Float2Int64:
	
	break;
    case infini::CastType::Float2Int32:break;
    case infini::CastType::Float2Int16:break;
    case infini::CastType::Float2Int8:break;
    case infini::CastType::Float2BFloat16:break;
    case infini::CastType::Int322Float:break;
    case infini::CastType::Int322Int8:break;
    case infini::CastType::Int322Int16:break;
    case infini::CastType::Int322Int64:break;
    case infini::CastType::Int162Float:break;
    case infini::CastType::Int162Int32:break;
    case infini::CastType::Int82Float:break;
    case infini::CastType::Int82Int16:break;
    case infini::CastType::Int82Int32:break;
    case infini::CastType::Uint82Float:break;
    case infini::CastType::Uint82Int32:break;
    case infini::CastType::Uint82Int64:break;
    case infini::CastType::Int642Int32:break;
    case infini::CastType::Int642Uint32:break;
    case infini::CastType::Int642Float:break;
    case infini::CastType::Uint322Int64:break;
    case infini::CastType::Float162Float:break;
    case infini::CastType::BFloat162Float:break;
    case infini::CastType::Float2Float:break;
		}
        return std::nullopt;
    }

    std::string CastObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

    DataType CastObj::getOutputDataType() const
    {
        switch (castType)
        {
        case CastType::Float2Float16:
            return DataType::Float16;
        case CastType::Float2Int64:
            return DataType::Int64;
        case CastType::Float2Int32:
            return DataType::Int32;
        case CastType::Float2Int16:
            return DataType::Int16;
        case CastType::Float2Int8:
            return DataType::Int8;
        case CastType::Int322Float:
            return DataType::Float32;
        case CastType::Int322Int8:
            return DataType::Int8;
        case CastType::Int322Int16:
            return DataType::Int16;
        case CastType::Int162Float:
            return DataType::Float32;
        case CastType::Int162Int32:
            return DataType::Int32;
        case CastType::Int82Float:
            return DataType::Float32;
        case CastType::Int82Int16:
            return DataType::Int16;
        case CastType::Int82Int32:
            return DataType::Int32;
        case CastType::Uint82Float:
            return DataType::Float32;
        case CastType::Uint82Int32:
            return DataType::Int32;
        case CastType::Uint82Int64:
            return DataType::Int64;
        case CastType::Int322Int64:
            return DataType::Int64;
        case CastType::Int642Int32:
            return DataType::Int32;
        case CastType::Int642Uint32:
            return DataType::UInt32;
        case CastType::Int642Float:
            return DataType::Float32;
        case CastType::Uint322Int64:
            return DataType::Int64;
        case CastType::Float162Float:
            return DataType::Float32;
        case CastType::BFloat162Float:
            return DataType::Float32;
        case CastType::Float2BFloat16:
            return DataType::BFloat16;
        case CastType::Float2Float:
            return DataType::Float32;
        default:
            IT_TODO_HALT();
        }
    }
}; // namespace infini
