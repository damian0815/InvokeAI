import { Flex, Divider, Box, Image } from "@invoke-ai/ui-library";
import { memo, useCallback, useMemo, useRef, useState } from "react";
import { ImageComparisonDroppable } from "./ImageComparisonDroppable";
import { useImageDTO } from "services/api/endpoints/images";
import { useAppSelector } from "app/store/storeHooks";

import { selectImageToTokenize } from 'features/gallery/store/gallerySelectors';
import { useDebouncedMetadata } from "services/api/hooks/useDebouncedMetadata";
import { Dimensions } from "@xyflow/react";
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useStore } from '@nanostores/react';
import { fitDimsToContainer } from "./common";
import { TokenizationToolbar } from "./TokenizationToolbar";

export const ImageTokenization = memo(() => {

  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  const imageDTO = useImageDTO(useAppSelector(selectImageToTokenize));
  const { metadata, isLoading } = useDebouncedMetadata(imageDTO?.image_name);

  // Ref callback runs synchronously when the DOM node is attached, ensuring we have a measurement before
  // the tokenization content is rendered.
  const measureNode = useCallback((node: HTMLDivElement) => {
    if (node) {
      ref.current = node;
      const boundingRect = node.getBoundingClientRect();
      setRect(boundingRect);
    }
  }, []);

  const fittedDims = useMemo<Dimensions>(() => {
    if (!rect || !imageDTO) {
      return { width: 0, height: 0 };
    }
    return fitDimsToContainer(rect, imageDTO);
  }, [imageDTO, rect]);

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <TokenizationToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        <Box ref={measureNode} w="full" h="full" overflow="hidden">
          {imageDTO && metadata && metadata["attention_maps"] && metadata["tokenization"] && <ImageTokenizationContent image={imageDTO} metadata={metadata} fittedDims={fittedDims} />}
        </Box>
      </Flex>
    </Flex>
  );


});


const Tags = ({ tags }: { tags: string[] | undefined }) => {
  return <Box mb={2} maxH={24} overflowY="auto">
    <Flex gap={2} flexWrap="wrap">
      {tags && tags.map((tag, index) => <Box
        key={index}
        px={2}
        py={1}
        borderRadius="base"
        bg="base.200"
        color="base.800"
        fontSize="xs"
        maxW="max-content"
      >
        {tag}
      </Box>)}
    </Flex>
  </Box>;
}

const ImageTokenizationContent = memo(({ image, metadata, fittedDims }: { image: any; metadata: any; fittedDims: Dimensions }) => {
  const crossOrigin = useStore($crossOrigin);

  const attentionMapsDTO = useImageDTO(metadata["attention_maps"]["collection"][1]["image_name"]);

  const tokens = JSON.parse(metadata["tokenization"]["value"] || '[]');
  console.log(attentionMapsDTO, tokens)

  return <>
    <Tags tags={tokens['positive']} />
    <Image
      id="tokenization-image"
      src={image.image_url}
      fallbackSrc={image.thumbnail_url}
      crossOrigin={crossOrigin}
      w={fittedDims.width}
      h={fittedDims.height}
      maxW="full"
      maxH="full"
      objectFit="cover"
      objectPosition="top left"
    />
    {attentionMapsDTO && <Image
      id="tokenization-attention-map"
      src={attentionMapsDTO.image_url}
      fallbackSrc={attentionMapsDTO.thumbnail_url}
      crossOrigin={crossOrigin}
      w={fittedDims.width}
      h={fittedDims.height}
      maxW="full"
      maxH="full"
      objectFit="cover"
      objectPosition="top left"
    />}
    <pre>{JSON.stringify(metadata)}</pre>
  </>
});
ImageTokenization.displayName = 'ImageTokenization';
